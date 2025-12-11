import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from itertools import combinations
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import gamma, logsumexp
from scipy.stats import f,chi
from scipy.linalg import sqrtm
from scipy.integrate import cumulative_trapezoid
import os
#exponential mechanism


class ClusterNode:
    def __init__(self, points=None, left=None, right=None, distance=0, depth=0, parent = None):
        self.points = points  # Points contained in this cluster
        self.left = left  # Left child node (merged cluster)
        self.right = right  # Right child node (merged cluster)
        self.distance = distance  # Distance between merged clusters
        self.parent = parent
        self.depth = depth  # Depth of this node in the hierarchy

    def __repr__(self):
        return f"ClusterNode(points={self.points})"


class AgglomerativeClustering:
    def __init__(self, X, sigma = None, n_clusters=2, tau=1, affinity='euclidean', linkage='single', random_state=None):
        self.X = X
        self.sigma = sigma
        self.n = np.shape(X)[0]
        self.p = np.shape(X)[1]
        self.tau = tau
        self.cluster_nodes = None
        self.distance_matrix = None
        self.n_clusters = n_clusters  # Number of clusters to form
        self.affinity = affinity  # Distance metric
        self.linkage = linkage  # Linkage criteria
        self.root = None  # Root of the cluster hierarchy
        self.step = 0
        self.existing_clusters_log = {}
        # dictionary of all clusters that have ever existed to retrieve distance.
        # key: the winning clusters at the step.
        # item: all the existing clusters (before merge) at this step
        self.distance_log = {}
        # Dictionary saving all distances
        # key:
        # item:
        self.labels = []

        self.tau_t_log = []
        self.linkage_matrix = []
        # (n-1) x 4 matrix to draw dendrogram
        # id1, id2, randomized distance, # of points in the new cluster
        self.cluster_id_counter = self.n  # IDs for merged clusters start after sample indices
        self.node_to_id = {}

        # handling non-spherical data case

        #if np.isscalar(sigma) or sigma is None:
        #    self.Z = X
        #else:
        #    from scipy.linalg import sqrtm
        #    inv_sqrt = np.linalg.inv(sqrtm(sigma))
        #    self.Z = X @ inv_sqrt  # whitened feature space

        self.random_state = random_state
        if random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

    def fit(self, dendrogram = False):
        self.n_samples = self.X.shape[0]
        self.cluster_nodes = [ClusterNode(points=[i]) for i in
                              range(self.n_samples)]  #initial step: each point as a cluster
        for i, node in enumerate(self.cluster_nodes):
            self.node_to_id[node] = i

        self.distance_matrix = self._compute_distance_matrix()  #initial

        while len(self.cluster_nodes) > self.n_clusters:
            current_clusters = self.cluster_nodes.copy()
            self.step += 1
            # Find the two closest clusters
            i, j = self._find_winning_clusters(self.distance_matrix)
            #print("i",i)
            #print("j",j)
            self.existing_clusters_log[(self.cluster_nodes[i], self.cluster_nodes[j])] = current_clusters.copy()
            self._merge_clusters(i, j, self.distance_matrix)
            #print(self.distance_matrix)

        self.K_clusters = self.cluster_nodes.copy() #store the final K cluster

        if dendrogram and len(self.cluster_nodes) > 1:
            self._complete_dendrogram_construction()

    def _complete_dendrogram_construction(self):
        """
        Continue merging from current state until one root remains.
        This is only for dendrogram purposes and does not change cluster assignments.
        """
        while len(self.cluster_nodes) > 1:
            current_clusters = self.cluster_nodes.copy()
            self.step += 1
            i, j = self._find_winning_clusters(self.distance_matrix)
            self._merge_clusters(i, j, self.distance_matrix)
        self.root = self.cluster_nodes[0]
        self.final_step = self.step

    def _compute_distance_matrix(self, data=None):
        """Compute the initial distance matrix for all points."""
        if data is None:
            data = self.X
            #data = self.Z
        from scipy.spatial.distance import pdist, squareform
        distance_matrix = squareform(pdist(data, metric=self.affinity))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):  # Only upper triangular part
                self.distance_log[(self.cluster_nodes[i], self.cluster_nodes[j])] = distance_matrix[i, j]
        return distance_matrix

    def _find_winning_clusters(self, distance_matrix):
        """Find the indices of the two closest clusters.
            i,j = argmin d(G_i,G_j; X) + W(G_i,G_j)"""
        clusters_sorted = sorted(self.cluster_nodes, key=lambda c: min(c.points))
        closest_clusters = (-1, -1)
        scores = []
        pair_idxs = []
        Ds = [] #array storing the pairwise distance
        if self.tau!=0:
            for i in range(len(clusters_sorted)):
                for j in range(i + 1, len(clusters_sorted)):
                    cluster1, cluster2 = self.cluster_nodes[i], self.cluster_nodes[j]
                    idx = (i,j)
                    D_ij = self._calculate_linkage_distance(cluster1,cluster2,self.X)
                    Ds.append(D_ij)
                    pair_idxs.append(idx)
            tau_t = self.tau * np.mean(Ds)
            self.tau_t_log.append(tau_t)
            scores = [np.exp(-(1/tau_t) * D_ij) for D_ij in Ds]
            scores_norm = scores/np.sum(scores)
            index = range(len(pair_idxs))
            winning_cluster_idx = self.rng.choice(index, 1, p=scores_norm)[0]
            winning_cluster = pair_idxs[winning_cluster_idx]
            return winning_cluster
        else:
            min_distance = np.inf
            closest_clusters = (-1, -1)

            for i in range(len(self.cluster_nodes)):
                for j in range(i + 1, len(self.cluster_nodes)):
                    distance = distance_matrix[i, j]

                    if distance < min_distance:
                        min_distance = distance
                        closest_clusters = (i, j)
            return closest_clusters


    def _merge_clusters(self, i, j, distance_matrix, data=None):
        """Merge two clusters and update the distance matrix."""
        if data is None:
            #data = self.Z
            data = self.X

        # Merge clusters
        merged_points = self.cluster_nodes[i].points + self.cluster_nodes[j].points
        new_node = ClusterNode(points=merged_points, left=self.cluster_nodes[i], right=self.cluster_nodes[j],
                               distance=distance_matrix[i, j],
                               depth=max(self.cluster_nodes[i].depth, self.cluster_nodes[j].depth) + 1)
        self.cluster_nodes[i].parent = new_node
        self.cluster_nodes[j].parent = new_node

        self.cluster_nodes.append(new_node)

        # update linkage matrix for dendrogram
        new_node_id = self.cluster_id_counter
        self.node_to_id[new_node] = new_node_id
        self.cluster_id_counter += 1

        # Get child node IDs
        id1 = self.node_to_id[self.cluster_nodes[i]]
        id2 = self.node_to_id[self.cluster_nodes[j]]

        # Record the merge in the linkage matrix
        num_points = len(new_node.points)
        dist = new_node.distance
        self.linkage_matrix.append([id1, id2, dist, num_points])

        # Update the distance matrix
        self.distance_matrix = self._update_distance_matrix(distance_matrix, new_node, i, j, data)


        # Remove the merged clusters from the list
        self.cluster_nodes.pop(max(i, j))  # Remove the higher index first
        self.cluster_nodes.pop(min(i, j))  # Then remove the lower index



    def _update_distance_matrix(self, distance_matrix, new_node, i, j, data=None):
        """
        Update the distance matrix after merging clusters.
        All other entries stay the same, only need to update the distance related to new node
        """
        if data is None:
            # data = self.Z
            data = self.X

        new_size = distance_matrix.shape[0] + 1
        new_distance_matrix = np.zeros((new_size, new_size))

        new_distance_matrix[:new_size - 1, :new_size - 1] = distance_matrix

        # Compute new distances from the new node to all remaining clusters
        for k in range(len(self.cluster_nodes)):
            if k == i or k == j:
                continue
            # Compute distance between new_node and cluster k
            dist = self._calculate_linkage_distance(new_node, self.cluster_nodes[k], data)
            new_distance_matrix[new_size - 1, k] = dist
            new_distance_matrix[k, new_size - 1] = dist
            self.distance_log[(new_node, self.cluster_nodes[k])] = dist

        # Remove the old distances
        distance_matrix = np.delete(new_distance_matrix, (i, j), axis=0)
        distance_matrix = np.delete(distance_matrix, (i, j), axis=1)
        return distance_matrix

    def get_cluster_labels(self):
        """Extract cluster labels for each point."""
        labels = np.zeros(self.n_samples, dtype=int)
        for cluster_id, node in enumerate(self.K_clusters):
            for point in node.points:
                labels[point] = cluster_id
        self.labels = labels
        return labels

    def compute_wcss(self):
        """Compute the Within-Cluster Sum of Squares (WCSS) for the clustering."""
        labels = self.get_cluster_labels()
        wcss = 0
        for cluster in set(labels):
            # Extract points belonging to the current cluster
            cluster_points = self.X[labels == cluster]
            # Calculate the centroid of the cluster
            centroid = cluster_points.mean(axis=0)
            # Calculate the sum of squared distances of points to the centroid
            wcss += ((cluster_points - centroid) ** 2).sum()
        return wcss

    def compute_bcss(self):
        """Compute the Between-Cluster Sum of Squares (BCSS) for the clustering."""
        wcss = self.compute_wcss()
        overall_mean = np.mean(self.X, axis=0)
        tss = np.sum((self.X - overall_mean) ** 2)
        bcss = tss - wcss
        return bcss

    def plot_dendrogram(self, ax=None, show=True, outdir=None, save_fig=False, manual_color=False):
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
        import numpy as np, os

        tau = self.tau
        linkage_matrix = np.array(self.linkage_matrix)
        K = self.n_clusters
        cut_index = len(linkage_matrix) - K
        cut_height = linkage_matrix[cut_index, 2]

        clusters_at_K = self.K_clusters

        def clusters_to_labels(clusters, n_samples):
            labels = np.empty(n_samples, dtype=int)
            for cid, cluster in enumerate(clusters):
                for idx in cluster.points:
                    labels[idx] = cid
            return labels

        cluster_labels = clusters_to_labels(clusters_at_K, n_samples=self.n)
        cluster_labels = cluster_labels - cluster_labels.min() + 1

        if manual_color:
            if self.n_clusters == 3:
                target_points = [0, 10, 23]
            elif self.n_clusters == 2:
                target_points = [0, 10]
            target_colors = ["#9579d9", "#7ab13f", "#ff924c"]
            cluster_color_map = {cluster_labels[i]: c for i, c in zip(target_points, target_colors)}

            all_clusters = np.unique(cluster_labels)
            unused_colors = [c for c in target_colors if c not in cluster_color_map.values()]
            for c in all_clusters:
                if c not in cluster_color_map:
                    cluster_color_map[c] = unused_colors.pop(0) if unused_colors else "gray"

            def link_color_func(node_id):
                if node_id < len(cluster_labels):
                    return cluster_color_map[cluster_labels[node_id]]
                else:
                    left_child = int(linkage_matrix[node_id - len(cluster_labels), 0])
                    right_child = int(linkage_matrix[node_id - len(cluster_labels), 1])
                    left_color = link_color_func(left_child)
                    right_color = link_color_func(right_child)
                    return left_color if left_color == right_color else "gray"
        else:
            link_color_func = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        dendrogram(
            linkage_matrix,
            ax=ax,
            color_threshold=0 if manual_color else cut_height,
            above_threshold_color="gray" if manual_color else "C0",
            link_color_func=link_color_func,
        )

        ax.set_title(f" Randomized Hierarchical Clustering Dendrogram (tau = {self.tau})", fontsize=12)
        ax.set_xlabel("Sample Index", fontsize=12, fontweight='bold')
        ax.set_ylabel("Distance", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_xticklabels():
            label.set_rotation(0)

        if show and ax is None:
            plt.show()

        if save_fig and outdir is not None:
            plt.savefig(os.path.join(outdir, f"dendro_{tau}.png"))
            plt.close()

    def _calculate_linkage_distance(self, new_node, cluster, data=None):
        """Calculate the distance between clusters based on the chosen linkage method."""
        if data is None:
            # data = self.Z
            data = self.X

        if self.linkage == 'ward':
            return self._ward_distance(new_node, cluster, data)
        elif self.linkage == 'single':
            return self._single_linkage(new_node, cluster, data)
        elif self.linkage == 'complete':
            return self._complete_linkage(new_node, cluster, data)
        elif self.linkage == 'average':
            return self._average_linkage(new_node, cluster, data)
        elif self.linkage == 'weighted':
            return self._weighted_linkage(new_node, cluster, data)
        elif self.linkage == 'centroid':
            return self._centroid_linkage(new_node, cluster, data)
        elif self.linkage == 'median':
            return self._median_linkage(new_node, cluster, data)
        elif self.linkage == 'minimax':
            return self._minimax_linkage(new_node, cluster, data)
        else:
            raise ValueError("Unknown linkage method: {}".format(self.linkage))

    def _ward_distance(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X

        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)

        # Calculate the number of points in each cluster
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)

        # Calculate the squared distance between the centroids
        distance_between_centroids = np.sum((centroid_new - centroid_cluster) ** 2)

        # Calculate the Ward's distance: increase in variance after merging
        ward_distance = distance_between_centroids * (size_new * size_cluster) / (size_new + size_cluster)

        return float(ward_distance)

    def _single_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        # Single linkage: Minimum distance between clusters
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.min(distances))

    def _complete_linkage(self, new_node, cluster, data=None):
        # Complete linkage: Maximum distance between clusters
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.max(distances))

    def _average_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        distances = distance.cdist(data_new_node, data_cluster, metric=self.affinity)
        return float(np.mean(distances))

    def _minimax_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X

        all_points_idx = np.concatenate([new_node.points, cluster.points])
        data_all = data[all_points_idx]
        pairwise = distance.cdist(data_all, data_all, metric=self.affinity)
        radii = np.max(pairwise, axis=1)
        minimax_distance = np.min(radii)

        return float(minimax_distance)

    def _weighted_linkage(self, new_node, cluster, data=None):
        if data is None:
            data = self.X

        # Ensure neither cluster is empty
        size_new = len(new_node.points)
        size_cluster = len(cluster.points)
        if size_new == 0 or size_cluster == 0:
            raise ValueError("One of the clusters is empty.")

        # Compute centroids
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)

        dist = self._calculate_distance(centroid_new, centroid_cluster)
        return float(dist)

    def _centroid_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        centroid_new = np.mean(data_new_node, axis=0)
        centroid_cluster = np.mean(data_cluster, axis=0)
        return self._calculate_distance(centroid_new, centroid_cluster)

    def _median_linkage(self, new_node, cluster, data=None):
        if data is None:
            # data = self.Z
            data = self.X
        data_new_node = data[new_node.points]
        data_cluster = data[cluster.points]
        median_new = np.median(data_new_node, axis=0)
        median_cluster = np.median(data_cluster, axis=0)
        return self._calculate_distance(median_new, median_cluster)

    def _calculate_distance(self, point1, point2):
        """Calculate the distance between two points based on the chosen affinity."""
        if self.affinity == 'euclidean':
            return float(np.linalg.norm(point1 - point2))
        else:
            raise ValueError("Unknown affinity: {}".format(self.affinity))

    ### Inference part:

    def get_all_winning_pairs(self):
        winning_pairs = []
        dictionary = self.existing_clusters_log
        for idx, key in enumerate(dictionary.keys()):
            winning_pairs.append(key)
        return winning_pairs

    def compute_nu(self,node):
            # return the projection direction from the given node
        G_1 = np.array(node.left.points)
        G_2 = np.array(node.right.points)
        n_G1 = len(G_1)
        n_G2 = len(G_2)

        nu = np.zeros(self.n)

        nu[G_1] += 1 / n_G1
        nu[G_2] -= 1 / n_G2
        return nu

    def _sel_correction_F(self, node, grid, P2, R0, R1, S):

        # node: a ClusterNode saving point, left, right, distance between merged, depth
        # grid: each value is a grid value

        def find_current_step(node1,node2):
            dictionary = self.existing_clusters_log
            for idx, key in enumerate(dictionary.keys()):
                if (key == (node1,node2)) or (key == (node2,node1)):
                    return idx + 1
            return -1

        #get the parent clusters of the given node
        p_node_1 = node.left
        p_node_2 = node.right
        m = len(p_node_1.points) + len(p_node_2.points)
        nu = self.compute_nu(node)
        nu_norm = np.linalg.norm(nu)
        #print("m: ",m)
        current_step = find_current_step(p_node_1, p_node_2)
        #print("current step: {}".format(current_step))
        all_winning_pairs = self.get_all_winning_pairs()
        #print("all winning pairs: {}".format(all_winning_pairs))

        cor_prob = np.zeros_like(grid) #for each grid value, cor_prob[g] = \sum (p(\hat{s}^{(t)}|X(g)))
        G_w_1 = p_node_1  #G^{(t)}_1 and G^{(t)}_2
        G_w_2 = p_node_2
        s = current_step  #going from top level to the beginning
        corrections = np.zeros((len(grid),s))
        while s > 0:
            #print("level: ", s)
            merged_pair = (G_w_1, G_w_2)
            #print("winning pair at this step: ", merged_pair)
            merged_pair_r = (G_w_2, G_w_1)
            # to get all the existing cluster at this step
            if merged_pair in self.existing_clusters_log.keys():
                clusters_s = self.existing_clusters_log[merged_pair]
            else:
                clusters_s = self.existing_clusters_log[merged_pair_r]

            for g_idx, g in enumerate(grid):
                # get the reconstructed X_grid from grid value
                #print("grid value: ", g)
                cor_scores = [] #the vector [p_1,....,p_d], first item is always the optimal
                Ds_grid = []
                X_grid = (np.sqrt((g)/(m-2+(g))) * R0 + np.sqrt((m-2)/(m-2+(g))) * R1) *np.sqrt(S) + P2 @ self.X
                D_opt_grid = self._calculate_linkage_distance(G_w_1, G_w_2, X_grid)  #D(\hat{G}_1, \hat{G}_2; X_grid)
                Ds_grid.append(D_opt_grid)

                pairs = combinations(clusters_s, 2)
                for cluster1, cluster2 in pairs:
                    if not ((G_w_1 == cluster1 and G_w_2 == cluster2) or (G_w_2 == cluster1 and G_w_1 == cluster2)):
                        D_grid = self._calculate_linkage_distance(cluster1, cluster2, X_grid)
                        Ds_grid.append(D_grid)

                tau_t_grid = self.tau * np.mean(Ds_grid)
                cor_scores = [np.exp(-(1/tau_t_grid) *  D_grid) for D_grid in Ds_grid]
                cor_scores = (cor_scores / np.sum(cor_scores)) #normalization
                #cor_scores[0] = exp(-1\e*d(s_hat;X(u)))/ sum_s exp(-1/e*d(s;X(u))) = P(s_hat|X(u))
                cor_prob[g_idx] = np.log(cor_scores[0])
                #cor_prob[g_idx] += np.log(cor_scores[0])
                #print("cor_prob: ", cor_prob[g_idx])

            corrections[:,s-1] += cor_prob

            if s>1:
                winning_pair_s = all_winning_pairs[s - 2] #get the winning pair of previous level
                G_w_1 = winning_pair_s[0]
                G_w_2 = winning_pair_s[1]

            s -= 1
        return np.array(corrections)
        #return np.array(cor_prob)
    def merge_inference_F(self, node, ngrid = 10000, ncoarse = 20, grid_width = 15):
        def create_indicator_diagonal_matrix(index_list, n):
            diag = np.zeros(n)
            diag[index_list] = 1
            return np.diag(diag), diag

        if self.tau!=0:
            nu = self.compute_nu(node).reshape(-1,1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points)+len(p_node_2.points)
            if m==2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan

            else:
                P0 = nu@nu.T / np.linalg.norm(nu)**2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points,self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points,self.n)
                one1 = one1.reshape(-1,1)
                one2 = one2.reshape(-1,1)
                P1 = (I1 - one1@one1.T / len(p_node_1.points)) + (I2 - one2@one2.T / len(p_node_2.points))
                P2 = np.eye(self.n) - P0 - P1

                S = np.linalg.norm(P0 @ self.X, 'fro')**2 + np.linalg.norm(P1 @ self.X, 'fro')**2
                R0 = (P0 @ self.X) / np.linalg.norm(P0 @ self.X, 'fro')
                R1 = (P1 @ self.X) / np.linalg.norm(P1 @ self.X, 'fro')

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / (np.linalg.norm(P1 @ self.X,'fro') ** 2)
                #print(np.linalg.norm(P0 @ self.X, 'fro') ** 2)
                #print(np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                #print(observed_target)
                #print("Are they close?", np.allclose(self.X, (np.sqrt(observed_target/(m-2+observed_target)) * R0 + np.sqrt((m-2)/(m-2+observed_target)) * R1) *np.sqrt(S) + P2 @ self.X))
                #projection_error = np.linalg.norm((np.eye(self.n) - np.outer(nu, nu) / np.linalg.norm(nu) ** 2) @ nu)
                #print("Projection error (should be close to 0):", projection_error)
                #print("obs:",observed_target)
                if ncoarse is not None:
                    coarse_grid = np.linspace(0.00001, grid_width, ncoarse)
                    eval_grid = coarse_grid
                else:
                    eval_grid = stat_grid

                if ncoarse is None:
                    sel_probs = self._sel_correction_F(node,stat_grid,P2, R0, R1, S)
                    p = self.p
                    log_prior = np.zeros(ngrid)
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=stat_grid[g], dfn=p, dfd=(m - 2) * p)
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)

                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if stat_grid[g] >= observed_target:
                            num += posterior[g]
                    p_value = num/sum
                else:
                    sel_probs_coarse = self._sel_correction_F(node,eval_grid,P2, R0, R1, S)
                    step = sel_probs_coarse.shape[1]

                    grid = np.linspace(0.00001, grid_width, num=ngrid)
                    sel_probs = np.zeros(ngrid)
                    log_prior = np.zeros(ngrid)
                    p = self.p


                    '''
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                    for s in range(step):
                        approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse[:,s],
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                        #for g in range(ngrid):
                            #sel_probs[g] += approx_fn(grid[g]) #selection probability
                    '''
                    log_prior = f.logpdf(x=grid, dfn=p, dfd=(m - 2) * p)

                    interpolation = np.array([
                        interp1d(eval_grid, sel_probs_coarse[:,s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(grid)
                        for s in range(step)
                    ])
                    sel_probs = interpolation.sum(axis=0)
                    '''
                    approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse,
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                                         
                                        for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                        sel_probs[g] = approx_fn(grid[g])
                    '''
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)


                    posterior = posterior / np.max(posterior)
                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if grid[g] >= (observed_target):
                            num += posterior[g]
                    p_value = num/sum
        else:
            nu = self.compute_nu(node).reshape(-1, 1)
            p_node_1 = node.left
            p_node_2 = node.right
            m = len(p_node_1.points) + len(p_node_2.points)
            if m==2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan
            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1,1)
                one2 = one2.reshape(-1,1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / np.linalg.norm(P1 @ self.X,
                                                                                                     'fro') ** 2

                sel_probs = 0
                p = self.p
                posterior = np.zeros(ngrid)
                for g in range(ngrid):
                    posterior[g] = f.pdf(stat_grid[g],p,(m-2)*p)

                sum = 0
                num = 0
                for g in range(ngrid):
                    sum += posterior[g]
                    if stat_grid[g] >= observed_target:
                        num += posterior[g]
                p_value = num / sum

        return (p_value, observed_target, sel_probs)

    def compute_nu_pair(self, c1,c2):
        # return the projection direction from the given node
        G_1 = np.array(c1.points)
        G_2 = np.array(c2.points)
        n_G1 = len(G_1)
        n_G2 = len(G_2)

        nu = np.zeros(self.n)

        nu[G_1] += 1 / n_G1
        nu[G_2] -= 1 / n_G2
        return nu
    def _sel_correction_F_random_pair(self, c1, c2,grid, P2, R0, R1, S):

        # node: a ClusterNode saving point, left, right, distance between merged, depth
        # grid: each value is a grid value

        def find_current_step(node1, node2):
            dictionary = self.existing_clusters_log
            for idx, key in enumerate(dictionary.keys()):
                if (key == (node1, node2)) or (key == (node2, node1)):
                    return idx + 1
            return -1

        # get the parent clusters of the given node
        p_node_1 = c1
        p_node_2 = c2

        winning_pair = list(self.existing_clusters_log.keys())[-1]
        m = len(p_node_1.points) + len(p_node_2.points)
        nu = self.compute_nu_pair(c1,c2)
        nu_norm = np.linalg.norm(nu)
        # print("m: ",m)
        current_step = find_current_step(winning_pair[0], winning_pair[1])
        # print("current step: {}".format(current_step))
        all_winning_pairs = self.get_all_winning_pairs()
        # print("all winning pairs: {}".format(all_winning_pairs))

        cor_prob = np.zeros_like(grid)  # for each grid value, cor_prob[g] = \sum (p(\hat{s}^{(t)}|X(g)))
        G_w_1 = winning_pair[0] # G^{(t)}_1 and G^{(t)}_2
        G_w_2 = winning_pair[1]
        s = current_step  # going from top level to the beginning
        corrections = np.zeros((len(grid), s))
        while s > 0:
            # print("level: ", s)
            merged_pair = (G_w_1, G_w_2)
            # print("winning pair at this step: ", merged_pair)
            merged_pair_r = (G_w_2, G_w_1)
            # to get all the existing cluster at this step
            if merged_pair in self.existing_clusters_log.keys():
                clusters_s = self.existing_clusters_log[merged_pair]
            else:
                clusters_s = self.existing_clusters_log[merged_pair_r]

            for g_idx, g in enumerate(grid):
                # get the reconstructed X_grid from grid value
                # print("grid value: ", g)
                cor_scores = []  # the vector [p_1,....,p_d], first item is always the optimal
                Ds_grid = []
                X_grid = (np.sqrt((g) / (m - 2 + (g))) * R0 + np.sqrt((m - 2) / (m - 2 + (g))) * R1) * np.sqrt(
                    S) + P2 @ self.X
                D_opt_grid = self._calculate_linkage_distance(G_w_1, G_w_2, X_grid)  # D(\hat{G}_1, \hat{G}_2; X_grid)
                Ds_grid.append(D_opt_grid)

                pairs = combinations(clusters_s, 2)
                for cluster1, cluster2 in pairs:
                    if not ((G_w_1 == cluster1 and G_w_2 == cluster2) or (G_w_2 == cluster1 and G_w_1 == cluster2)):
                        D_grid = self._calculate_linkage_distance(cluster1, cluster2, X_grid)
                        Ds_grid.append(D_grid)

                tau_t_grid = self.tau * np.mean(Ds_grid)
                cor_scores = [np.exp(-(1 / tau_t_grid) * D_grid) for D_grid in Ds_grid]
                cor_scores = (cor_scores / np.sum(cor_scores))  # normalization
                # cor_scores[0] = exp(-1\e*d(s_hat;X(u)))/ sum_s exp(-1/e*d(s;X(u))) = P(s_hat|X(u))
                cor_prob[g_idx] = np.log(cor_scores[0])
                # cor_prob[g_idx] += np.log(cor_scores[0])
                # print("cor_prob: ", cor_prob[g_idx])

            corrections[:, s - 1] += cor_prob

            if s > 1:
                winning_pair_s = all_winning_pairs[s - 2]  # get the winning pair of previous level
                G_w_1 = winning_pair_s[0]
                G_w_2 = winning_pair_s[1]

            s -= 1
        return np.array(corrections)
        # return np.array(cor_prob)

    def merge_inference_F_random_pair(self, c1,c2, ngrid=10000, ncoarse=20, grid_width=15):
        def create_indicator_diagonal_matrix(index_list, n):
            diag = np.zeros(n)
            diag[index_list] = 1
            return np.diag(diag), diag

        if self.tau != 0:
            nu = self.compute_nu_pair(c1,c2).reshape(-1, 1)
            p_node_1 = c1
            p_node_2 = c2
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan

            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))
                P2 = np.eye(self.n) - P0 - P1

                S = np.linalg.norm(P0 @ self.X, 'fro') ** 2 + np.linalg.norm(P1 @ self.X, 'fro') ** 2
                R0 = (P0 @ self.X) / np.linalg.norm(P0 @ self.X, 'fro')
                R1 = (P1 @ self.X) / np.linalg.norm(P1 @ self.X, 'fro')

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / (
                            np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                # print(np.linalg.norm(P0 @ self.X, 'fro') ** 2)
                # print(np.linalg.norm(P1 @ self.X, 'fro') ** 2)
                # print(observed_target)
                # print("Are they close?", np.allclose(self.X, (np.sqrt(observed_target/(m-2+observed_target)) * R0 + np.sqrt((m-2)/(m-2+observed_target)) * R1) *np.sqrt(S) + P2 @ self.X))
                # projection_error = np.linalg.norm((np.eye(self.n) - np.outer(nu, nu) / np.linalg.norm(nu) ** 2) @ nu)
                # print("Projection error (should be close to 0):", projection_error)
                # print("obs:",observed_target)
                if ncoarse is not None:
                    coarse_grid = np.linspace(0.00001, grid_width, ncoarse)
                    eval_grid = coarse_grid
                else:
                    eval_grid = stat_grid

                if ncoarse is None:
                    sel_probs = self._sel_correction_F_random_pair(c1,c2, stat_grid, P2, R0, R1, S)
                    p = self.p
                    log_prior = np.zeros(ngrid)
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=stat_grid[g], dfn=p, dfd=(m - 2) * p)
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)

                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if stat_grid[g] >= observed_target:
                            num += posterior[g]
                    p_value = num / sum
                else:
                    sel_probs_coarse = self._sel_correction_F_random_pair(c1, c2, eval_grid, P2, R0, R1, S)
                    step = sel_probs_coarse.shape[1]

                    grid = np.linspace(0.00001, grid_width, num=ngrid)
                    p = self.p

                    '''
                    for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                    for s in range(step):
                        approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse[:,s],
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')
                        #for g in range(ngrid):
                            #sel_probs[g] += approx_fn(grid[g]) #selection probability
                    '''
                    log_prior = f.logpdf(x=grid, dfn=p, dfd=(m - 2) * p)

                    interpolation = np.array([
                        interp1d(eval_grid, sel_probs_coarse[:, s],
                                 kind='quadratic',
                                 bounds_error=False,
                                 fill_value='extrapolate')(grid)
                        for s in range(step)
                    ])
                    sel_probs = interpolation.sum(axis=0)
                    '''
                    approx_fn = interp1d(eval_grid,
                                         sel_probs_coarse,
                                         kind='quadratic',
                                         bounds_error=False,
                                         fill_value='extrapolate')

                                        for g in range(ngrid):
                        log_prior[g] = f.logpdf(x=grid[g], dfn=p, dfd=(m - 2) * p)
                        sel_probs[g] = approx_fn(grid[g])
                    '''
                    log_post = log_prior + sel_probs
                    posterior = np.exp(log_post)

                    posterior = posterior / np.max(posterior)
                    sum = 0
                    num = 0
                    for g in range(ngrid):
                        sum += posterior[g]
                        if grid[g] >= (observed_target):
                            num += posterior[g]
                    p_value = num / sum
        else:
            nu = self.compute_nu_pair(c1,c2).reshape(-1, 1)
            p_node_1 = c1
            p_node_2 = c2
            m = len(p_node_1.points) + len(p_node_2.points)
            if m == 2:
                p_value = np.nan
                observed_target = np.nan
                sel_probs = np.nan
            else:
                P0 = nu @ nu.T / np.linalg.norm(nu) ** 2
                I1, one1 = create_indicator_diagonal_matrix(p_node_1.points, self.n)
                I2, one2 = create_indicator_diagonal_matrix(p_node_2.points, self.n)
                one1 = one1.reshape(-1, 1)
                one2 = one2.reshape(-1, 1)
                P1 = (I1 - one1 @ one1.T / len(p_node_1.points)) + (I2 - one2 @ one2.T / len(p_node_2.points))

                stat_grid = np.linspace(0.00001, grid_width, num=ngrid)
                observed_target = (m - 2) * np.linalg.norm(P0 @ self.X, 'fro') ** 2 / np.linalg.norm(P1 @ self.X,
                                                                                                     'fro') ** 2

                sel_probs = 0
                p = self.p
                posterior = np.zeros(ngrid)
                for g in range(ngrid):
                    posterior[g] = f.pdf(stat_grid[g], p, (m - 2) * p)

                sum = 0
                num = 0
                for g in range(ngrid):
                    sum += posterior[g]
                    if stat_grid[g] >= observed_target:
                        num += posterior[g]
                p_value = num / sum

        return (p_value, observed_target, sel_probs)
