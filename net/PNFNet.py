import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from backbone.pvtv2 import pvt_v2_b2
import numpy as np
import functools
from torch.nn import init
import torch.optim as optim
from torch.nn import Parameter as P
from torch.autograd import Variable
# import faiss
from sklearn.cluster import KMeans
from torch.distributions import Categorical

class MemoryConceptAttentionProto(nn.Module):
    """concept attention"""
    def __init__(self, ch, which_conv, pool_size_per_cluster, num_k, feature_dim, warmup_total_iter=1000, cp_momentum=0.3, \
                cp_phi_momentum=0.95, normalization="together"):
        super(MemoryConceptAttentionProto, self).__init__()
        self.myid = "atten_concept_prototypes"
        self.pool_size_per_cluster = pool_size_per_cluster
        self.num_k = num_k
        self.feature_dim = feature_dim
        self.ch = ch  # input channel
        self.total_pool_size = self.num_k * self.pool_size_per_cluster

        self.register_buffer('concept_pool', torch.rand(self.feature_dim, self.total_pool_size))
        self.register_buffer('concept_proto', torch.rand(self.feature_dim, self.num_k))

        self.register_buffer('warmup_iter_counter', torch.FloatTensor([0.]))
        self.warmup_total_iter = warmup_total_iter
        self.register_buffer('pool_structured', torch.FloatTensor([0.]))  # 0 means pool is un clustered, 1 mean pool is structured as clusters arrays

        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi_k = nn.ModuleList([self.which_conv(self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)])
        
        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)  # initialize
            param_phi_k.requires_grad = False  # not update by gradient

        self.g = self.which_conv(self.ch, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.feature_dim, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

        # self.momentum
        self.cp_momentum = cp_momentum
        self.cp_phi_momentum = cp_phi_momentum

    @torch.no_grad()
    def _update_pool(self, index, content):
        """update concept pool according to the content
        index: [m, ]
        content: [c, m]
        """
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        
        # print("Updating concept pool...")
        self.concept_pool[:, index] = content.clone()
    
    @torch.no_grad()
    def _update_prototypes(self, index, content):
        assert len(index.shape) == 1
        assert content.shape[1] == index.shape[0]
        assert content.shape[0] == self.feature_dim
        # print("Updating prototypes...")
        self.concept_proto[:, index] = content.clone()

    @torch.no_grad()
    def computate_prototypes(self):
        """compute prototypes based on current pool"""
        assert not self._get_warmup_state(), f"still in warm up state {self.warmup_state}, computing prototypes is forbidden"
        self.concept_proto = self.concept_pool.detach().clone().reshape(self.feature_dim, self.num_k, self.pool_size_per_cluster).mean(2)
    
    @torch.no_grad()
    def forward_update_pool(self, activation, cluster_num, momentum=None):

        if not momentum:
            momentum = 1.
        
        assert cluster_num.max() < self.num_k

        index = cluster_num * self.pool_size_per_cluster + torch.randint(self.pool_size_per_cluster, size=(cluster_num.shape[0],)).to(activation.device)
        
        self.concept_pool[:, index] = (1. - momentum) * self.concept_pool[:, index].clone() + momentum * activation.detach().T
        
    @torch.no_grad()
    def pool_kmean_init_gpu(self, seed=0, gpu_num=0, temperature=1):
        
        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 100
        clus.nredo = 10
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_num   
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        

        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        print(density.mean())
        density = temperature*density/density.mean()  #scale the mean to temperature 
        
        centroids = torch.Tensor(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster)             
        density = torch.Tensor(density)
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
        del cfg, res, index, clus

        self.structure_memory_bank(results) 
        print("Finish kmean init...")
        del results
    
    @torch.no_grad()
    def pool_kmean_init(self, seed=0, gpu_num=0, temperature=1):
        """TODO: clear up
        perform kmeans for cluster concept pool initialization
        Args:
            x: data to be clustered
        """
        
        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]}
        x = self.concept_pool.clone().cpu().numpy().T
        x = np.ascontiguousarray(x)
        num_cluster = self.num_k
        
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x)

        centroids = torch.Tensor(kmeans.cluster_centers_)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)   
        im2cluster = torch.LongTensor(kmeans.labels_)

        results['centroids'].append(centroids)
        results['im2cluster'].append(im2cluster)    
        
        self.structure_memory_bank(results) 
        print("Finish kmean init...")
    

    @torch.no_grad()
    def structure_memory_bank(self, cluster_results):
        """make memory bank structured """
        centeriod = cluster_results['centroids'][0] # [num_k, feature_dim]
        cluster_assignment = cluster_results['im2cluster'][0] # [total_pool_size,]
        
        mem_index = torch.zeros(self.total_pool_size).long() # array of memory index that contains instructions of how to rearange the memory into structured clusters
        memory_states = torch.zeros(self.num_k,).long() # 0 indicate the cluster has not finished structured
        memory_cluster_insert_ptr = torch.zeros(self.num_k,).long() # ptr to each cluster block

        for idx, i in enumerate(cluster_assignment):
            cluster_num = i
            if memory_states[cluster_num] == 0:
                
                mem_index[cluster_num * self.pool_size_per_cluster + memory_cluster_insert_ptr[cluster_num]] = idx  

                memory_cluster_insert_ptr[cluster_num] += 1
                if memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster:
                    memory_states[cluster_num] = 1 - memory_states[cluster_num]
            else:
                assert memory_cluster_insert_ptr[cluster_num] == self.pool_size_per_cluster
        
        not_fill_cluster = torch.where(memory_states == 0)[0]
        print(f"memory_states {memory_states}")
        print(f"memory_cluster_insert_ptr {memory_cluster_insert_ptr}")
        for unfill_cluster in not_fill_cluster:
            cluster_ptr = memory_cluster_insert_ptr[unfill_cluster]
            assert cluster_ptr != 0, f"cluster_ptr {cluster_ptr} is zero!!!"
            existed_index = mem_index[unfill_cluster * self.pool_size_per_cluster : unfill_cluster * self.pool_size_per_cluster + cluster_ptr]
            print(f"existed_index {existed_index}")
            print(f"cluster_ptr {cluster_ptr}")
            print(f"(self.pool_size_per_cluster {self.pool_size_per_cluster}")
            replicate_times = (self.pool_size_per_cluster // cluster_ptr) + 1 # with more replicate and cutoff
            print(f"replicate_times {replicate_times}")
            replicated_index = torch.cat([existed_index for _ in range(replicate_times)])
            print(f"replicated_index {replicated_index}")

            replicated_index = replicated_index[torch.randperm(replicated_index.shape[0])][:self.pool_size_per_cluster] # [pool_size_per_cluster, ]

            assert replicated_index.shape[0] == self.pool_size_per_cluster, f"replicated_index ({replicated_index.shape}) should has the same len as pool_size_per_cluster ({self.pool_size_per_cluster})"
            mem_index[unfill_cluster * self.pool_size_per_cluster: (unfill_cluster+1) * self.pool_size_per_cluster] = replicated_index

            memory_cluster_insert_ptr[unfill_cluster] = self.pool_size_per_cluster

            memory_states[unfill_cluster] = 1
        
        assert (memory_states == 0).sum() == 0, f"memory_states has zeros: {memory_states}"
        assert (memory_cluster_insert_ptr != self.pool_size_per_cluster).sum() == 0, f"memory_cluster_insert_ptr didn't match with pool_size_per_cluster: {memory_cluster_insert_ptr}"


        # update the real pool
        self._update_pool(torch.arange(mem_index.shape[0]), self.concept_pool[:, mem_index])
        # initialize the prototype
        self._update_prototypes(torch.arange(self.num_k), centeriod.T.to(self.concept_pool.device))
        print(f"Concept pool updated by kmeans clusters...")

    def _check_warmup_state(self):
        """check if need to switch warup_state to 0; when turn off warmup state, trigger k-means init for clustering"""
        # assert self._get_warmup_state(), "Calling _check_warmup_state when self.warmup_state is 0 (0 means not in warmup state)"
        
        if self.warmup_iter_counter == self.warmup_total_iter:
            # trigger kmean concept pool init
            self.pool_kmean_init()
        
    def warmup_sampling(self, x):
        """
        linearly sample input x to make it 
        x: [n, c, h, w]"""
        n, c, h, w = x.shape
        assert self._get_warmup_state(), "calling warmup sampling when warmup state is 0"
        
        # evenly distributed across space
        sample_per_instance = max(int(self.total_pool_size / n), 1)
        
        # sample index
        index = torch.randint(h * w, size=(n, 1, sample_per_instance)).repeat(1, c, 1).to(x.device) # n, c, sample_per_instance
        sampled_columns = torch.gather(x.reshape(n, c, h * w), 2, index) # n, c, sample_per_instance 
        sampled_columns = torch.transpose(sampled_columns, 1, 0).reshape(c, -1).contiguous() # c, n * sample_per_instance
        
        # calculate percentage to populate into pool, as the later the better, use linear intepolation from 1% to 50% according to self.warmup_iter_couunter
        percentage = (self.warmup_iter_counter + 1) / self.warmup_total_iter * 0.5 # max percent is 50%
        print(f"percentage {percentage.item()}")
        sample_column_num = max(1, int(percentage * sampled_columns.shape[1]))
        sampled_columns_idx = torch.randint(sampled_columns.shape[1], size=(sample_column_num,))
        sampled_columns = sampled_columns[:, sampled_columns_idx]  # [c, sample_column_num]

        # random select pool idx to update
        update_idx = torch.randperm(self.concept_pool.shape[1])[:sample_column_num]
        self._update_pool(update_idx, sampled_columns)

        # update number
        self.warmup_iter_counter += 1

    def forward(self, x, evaluation=False):
        device = x.device  # 确保在 forward 中动态地获取设备

        if self._get_warmup_state():
            print(f"Warmup state? {self._get_warmup_state()} self.warmup_iter_counter {self.warmup_iter_counter.item()} self.warmup_total_iter {self.warmup_total_iter}")
            # transform into low dimension
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)      # [n, c, h, w]
            g = self.g(x)          # [n, c, h, w]

            n, c, h, w = theta.shape

            # if still in warmup, skip attention
            self.warmup_sampling(phi) 
            self._check_warmup_state()

            # normal self attention
            theta = theta.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            phi = phi.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
            g = g.view(-1, self.feature_dim, x.shape[2] * x.shape[3])

            # Matmul and softmax to get attention maps
            beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1)

            # Attention map times g path
            o = self.o(torch.bmm(g, beta.transpose(1, 2).contiguous()).view(-1, self.feature_dim, x.shape[2], x.shape[3]))
            
            return self.gamma * o + x

        else:
            # transform into low dimension
            theta = self.theta(x)  # [n, c, h, w]
            phi = self.phi(x)
            g = self.g(x)          # [n, c, h, w]
            n, c, h, w = theta.shape

            # attend to concepts 
            ## selecting cooresponding prototypes -> [n, h, w]
            theta = torch.transpose(torch.transpose(theta, 0, 1).reshape(c, n * h * w), 0, 1).contiguous() # n * h * w, c
            phi = torch.transpose(torch.transpose(phi, 0, 1).reshape(c, n * h * w), 0, 1).contiguous() # n * h * w, c
            g = torch.transpose(torch.transpose(g, 0, 1).reshape(c, n * h * w), 0, 1).contiguous() # n * h * w, c
            with torch.no_grad():
                theta_atten_proto = nn.CosineSimilarity(dim=-1)(theta.unsqueeze(-2), self.concept_proto.detach().clone().T.unsqueeze(-3)) # n * h * w, num_k
                cluster_affinity = F.softmax(theta_atten_proto, dim=1) # n * h * w, num_k
                cluster_assignment = cluster_affinity.max(1)[1] # [n * h * w, ]

            dot_product = []
            cluster_indexs = []

            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0] # [n * h * w]
                theta_cluster = theta[cluster_index] # number of data  belong to the same cluster, c
                
                # attend to certain cluster
                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (cluster + 1) * self.pool_size_per_cluster] # [c, pool_size_per_cluster]
                
                theta_cluster_attend_weight = torch.matmul(theta_cluster, cluster_pool) # [num_data_in_cluster, pool_size_per_cluster]

                dot_product.append(theta_cluster_attend_weight)
                cluster_indexs.append(cluster_index)
            
            # integrate into one tensor
            dot_product = torch.cat(dot_product, axis=0) # [n * h * w, pool_size_per_cluster] but with different order
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            # remap it back into order Variable(torch.ones(2, 2), requires_grad=True)
            mapping_to_normal_index = torch.argsort(cluster_indexs)
            similarity_clusters = dot_product[mapping_to_normal_index] # n * h * w, pool_size_per_cluster
            
            # dot product with context
            similarity_context = torch.bmm(theta.reshape(n, h*w, c), torch.transpose(phi.reshape(n, h * w, c), 1, 2)) # [n, h*w, h*w]
            similarity_context = similarity_context.reshape(n * h * w, h * w) # n * h * w, h * w
            atten_weight = torch.cat([similarity_clusters, similarity_context], axis=1) # [n * h * w, pool_size_per_cluster + h * w]

            # attend 
            pool_residuals = []
            cluster_indexs = []
            for cluster in range(self.num_k):
                cluster_index = torch.where(cluster_assignment == cluster)[0] # [n * h * w]
                theta_cluster = theta[cluster_index] # number of data  belong to the same cluster, c
                atten_weight_pool_cluster = atten_weight[cluster_index, :self.pool_size_per_cluster] # [number of data  belong to the same cluster, pool_size_per_cluster]
                # softmax
                atten_weight_pool_cluster = F.softmax(atten_weight_pool_cluster, dim=1)
                
                # attend to certain cluster
                cluster_pool = self.concept_pool.detach().clone()[:, cluster * self.pool_size_per_cluster: (cluster + 1) * self.pool_size_per_cluster] # [c, pool_size_per_cluster]
                pool_residual = torch.matmul(atten_weight_pool_cluster, cluster_pool.T) # [num_batch_data_in_cluster, c]
                pool_residuals.append(pool_residual)
                cluster_indexs.append(cluster_index)
            pool_residuals = torch.cat(pool_residuals, axis=0) # [n * h * w, c] but with different order
            cluster_indexs = torch.cat(cluster_indexs, axis=0)

            # remap it back into order 
            mapping_to_normal_index = torch.argsort(cluster_indexs)
            pool_residuals = pool_residuals[mapping_to_normal_index] # n * h * w, c with correct order 
            pool_residuals = pool_residuals.reshape(n, h * w, c) # n, h * w, c

            # add with context 
            atten_weight_context = atten_weight[:, self.pool_size_per_cluster:] # [n * h * w, h * w]
            # softmax
            atten_weight_context = F.softmax(atten_weight_context, dim=1)
            atten_weight_context = atten_weight_context.reshape(n, h*w, h*w) # n, h*w, h*w
            context_residuals = torch.bmm(atten_weight_context, g.reshape(n, h * w, c)) # n, h * w, c, context residual is calcuated by g not phi

            # integrate context residual with pool residual
            beta_residual = pool_residuals + context_residuals  # n, h * w, c
            beta_residual = torch.transpose(beta_residual, 1, 2).reshape(n, c, h, w).contiguous()

            o = self.o(beta_residual)

            ### update pool
            with torch.no_grad():
                # moca update
                phi_k = self.phi_k[0](x) # [n, c, h, w]
                phi_k = torch.transpose(torch.transpose(phi_k, 0, 1).reshape(c, n * h * w), 0, 1).contiguous() # n * h * w, c
                phi_k_atten_proto = torch.matmul(phi_k, self.concept_proto.detach().clone()) # n * h * w, self.num_k
                phi_k_atten_proto = phi_k_atten_proto.reshape(n, h * w, self.num_k) # n, h * w, self.num_k
                cluster_affinity_phi_k = F.softmax(phi_k_atten_proto, dim=2) # n, h * w, self.num_k
                cluster_assignment_phi_k = cluster_affinity_phi_k.max(2)[1].reshape(n * h * w, ) # [n * h * w, ]
            
                # update pool first to allow contextual information 
                # should use the lambda to update concept pool momentumlly
                self.forward_update_pool(phi_k, cluster_assignment_phi_k, momentum=self.cp_momentum)

                # update prototypes
                self.computate_prototypes()

                # update phi_k
                for param_q, param_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
                    param_k.data = param_k.data * self.cp_phi_momentum + param_q.data * (1. - self.cp_phi_momentum)

            if evaluation:
                return o * self.gamma + x, cluster_affinity
                
            return o * self.gamma + x

    def get_cluster_num_index(self, idx):
        assert idx < self.total_pool_size
        return idx // self.pool_size_per_cluster
    

    def get_cluster_ptr(self, cluster_num):
        """get starting pointer for cluster_num"""
        assert cluster_num < self.num_k, f"cluster_num {cluster_num} out of bound (totally has {self.num_k} clusters)"
        return self.pool_size_per_cluster * cluster_num
    
    def _get_warmup_state(self):
        return self.warmup_iter_counter.cpu() <= self.warmup_total_iter

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class FCM(nn.Module):
    def __init__(self, inplanesl, inplanesc, planes):
        super(FCM, self).__init__()
        self.conv3_1 = ConvBNR(inplanesl+inplanesc, planes, 3)
        self.conv3_2 = ConvBNR(planes+inplanesc, planes, 3)
        self.conv3_3 = ConvBNR(planes, planes, 3)
        self.conv3_4 = ConvBNR(planes, planes, 3)
        self.conv3_5 = nn.Conv2d(planes, planes, 3, padding = 1)
        self.conv1_1 = Conv1x1(planes, planes)
        self.conv1_2 = Conv1x1(planes, planes)
        self.block1 = InvertedResidualBlock(planes//2, planes//2, expand_ratio=2)
        self.block2 = InvertedResidualBlock(planes//2, planes//2, expand_ratio=2)
        self.block3 = InvertedResidualBlock(planes//2, planes//2, expand_ratio=2)
        self.block4 = InvertedResidualBlock(planes//2, planes//2, expand_ratio=2)

        self.dconv3_0 = ConvBNR(inplanesc, planes, 3)
        self.dconv3_1 = ConvBNR(planes, planes, 3, dilation=2)
        self.dconv3_2 = ConvBNR(planes, planes, 3, dilation=3)
        self.dconv3_3 = ConvBNR(planes, planes, 3, dilation=4)
        self.dconv3_4 = Conv1x1(planes, planes)

        t = int(abs((log(planes, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, x, high):
        if low.size() != x.size():
            low = F.interpolate(low, x.size()[2:], mode='bilinear', align_corners=False)
        if high.size() != x.size():
            high = F.interpolate(high, x.size()[2:], mode='bilinear', align_corners=False)
        x_low = self.conv3_1(torch.cat((x, low), dim=1))
        x_high = self.conv3_2(torch.cat((x, high), dim=1))
        x_lowc = torch.chunk(x_low, 2, dim=1)
        x_low0 = self.block1(x_lowc[0]) * x_lowc[1]
        x_low1 = self.block2(x_lowc[1]) * x_lowc[0]
        x_low2 = self.block3(x_low0) + x_low1
        x_low3 = self.block4(x_low1) + x_low0
        l = self.conv1_2(torch.cat((x_low2, x_low3), dim=1))
        c = self.dconv3_0(x)
        c1 = self.dconv3_1(c)
        c2 = self.dconv3_2(c1 + c)
        c3 = self.dconv3_3(c2)
        c4 = self.dconv3_4(c3 + c2)
        wei = self.avg_pool(x_high)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        h = self.conv1_1(x_high * wei + x_high)
        lc = self.conv3_3(c4 + l)
        hc = self.conv3_4(c4 + h)
        out = torch.sigmoid(self.conv3_5(lc + hc))
        return out

class FCM2(nn.Module):
    def __init__(self, inplanesc, planes):
        super(FCM2, self).__init__()
        self.conv3_2 = ConvBNR(inplanesc+planes, planes, 3)
        self.conv3_5 = nn.Conv2d(planes, planes, 3, padding = 1)
        self.conv1_1 = Conv1x1(planes, planes)
        self.dconv3_0 = ConvBNR(inplanesc, planes, 3)
        self.dconv3_1 = ConvBNR(inplanesc, planes, 3, dilation=2)
        self.dconv3_2 = ConvBNR(planes, planes, 3, dilation=3)
        self.dconv3_3 = ConvBNR(planes, planes, 3, dilation=4)
        self.dconv3_4 = Conv1x1(planes, planes)
        t = int(abs((log(planes, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, high):
        if high.size() != x.size():
            high = F.interpolate(high, x.size()[2:], mode='bilinear', align_corners=False)
        x_high = self.conv3_2(torch.cat((x, high), dim=1))
        c = self.dconv3_0(x)
        c1 = self.dconv3_1(c)
        c2 = self.dconv3_2(c1 + c)
        c3 = self.dconv3_3(c2)
        c4 = self.dconv3_4(c3 + c2)
        wei = self.avg_pool(x_high)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        h = self.conv1_1(x_high * wei + x_high)
        out = torch.sigmoid(self.conv3_5(c4 + h))
        return out

class Conv_Block(nn.Module):  # [64, 128, 320, 512]
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(512+320+128, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample = ConvBNR(channel, channel, 3)
        self.cat_conv3 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.cat_conv5 = nn.Conv2d(channel*2, channel, kernel_size=5, stride=1, padding=2)
        self.cat_conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.param_free_norm = nn.BatchNorm2d(channel, affine=False)
        self.mlp_gamma = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x_low, x_high):
        if x_low.size() != x_high.size():
            x_high = F.interpolate(x_high, x_low.size()[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat((x_low, x_high), dim=1)
        x_cat3 = self.relu(self.cat_conv3(x_cat))
        x_cat5 = self.relu(self.cat_conv5(x_cat))
        x35 = self.relu(self.cat_conv3_2(x_cat3 * x_cat5 + x_low))
        normalized = self.param_free_norm(x35)
        ahla = self.mlp_gamma(normalized)
        beta = self.mlp_beta(normalized)
        out = x_high*ahla + x_low + beta
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = pvt_v2_b2()  
        path = '/home/lsl/lsl-IML/TCSVT2026/model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.fcm1 = FCM2(64, 64)
        self.fcm2 = FCM(64, 128, 64)
        self.fcm3 = FCM(128, 320, 64)
        self.fcm4 = FCM(320, 512, 64)
        self.ff1 = Fusion(64)
        self.ff2 = Fusion(64)
        self.ff3 = Fusion(64)
        self.ff4 = Fusion(64)
        self.conv_block = Conv_Block(64)
        self.ef = MemoryConceptAttentionProto(
            64, nn.Conv2d, pool_size_per_cluster=100, 
            num_k=10, feature_dim=128, warmup_total_iter=2000, cp_momentum=0.3, cp_phi_momentum=0.95
        )
        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(64, 1, 1)
        self.predictor4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x) # [64, 128, 320, 512, 512] [88,88  44,44  22,22 11,11, 22,22]
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        if x4.size()[2:] != x3.size()[2:]:
            x41 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear')
        if x2.size()[2:] != x3.size()[2:]:
            x21 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear')
        x5 = self.conv_block(x41, x3, x21)
        x5_ef = self.ef(x5)
        fc4 = self.fcm4(x3, x4, x5_ef)
        fc3 = self.fcm3(x2, x3 , fc4)
        fc2 = self.fcm2(x1, x2, fc3)
        fc1 = self.fcm1(x1, fc2)
        x45 = self.ff4(fc4, x5_ef)
        x345 = self.ff3(fc3, x45)
        x2345 = self.ff2(fc2, x345)
        x12345 = self.ff1(fc1, x2345)
        o3 = self.predictor4(x45)
        o3 = F.interpolate(o3, size=image_shape, mode='bilinear', align_corners=False)
        o2 = self.predictor3(x345)
        o2 = F.interpolate(o2, size=image_shape, mode='bilinear', align_corners=False)
        o1 = self.predictor2(x2345)
        o1 = F.interpolate(o1, size=image_shape, mode='bilinear', align_corners=False)
        o0 = self.predictor1(x12345)
        o0 = F.interpolate(o0, size=image_shape, mode='bilinear', align_corners=False)
        return o0, o1, o2, o3
