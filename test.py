
class PathTarget():
    def __init__(self, node_dict, orient_mat=None, merge=False, **kwargs):
        self.node_dict = node_dict
        self.orient_mat = orient_mat
        if merge:
            self.merge_data = self.merge_node(self.node_dict, **kwargs)
        else:
            self.merge_data = None

    def run_distribute_test(self, alpha=0.1, mode='ad', exp=True):
        if len(self.node_dict) > 2:
            raise ValueError('The members are more than 2.')
        group_A, group_B = self.node_dict.values()
        group_A = group_A.node_data['exp']
        group_B = group_B.node_data['exp']
        result_compare = []
        if not group_A.columns.equals(group_B.columns):
            raise ValueError('The features should be consistent.')
        for i in range(group_A.shape[1]):
            f_n = group_A.columns.values[i]
            A_i = np.expm1(group_A.iloc[:, i]) if exp else group_A.iloc[:, i]
            B_i = np.expm1(group_B.iloc[:, i]) if exp else group_B.iloc[:, i]
            if A_i.max() == 0 and B_i.max() == 0:
                continue
            else:
                if mode == 'ks':
                    statistic_, pvalue_ = ks_2samp(
                        group_A.iloc[:, i], group_B.iloc[:, i])
                elif mode == 'ad':
                    statistic_, _, pvalue_ = anderson_ksamp(
                        [group_A.iloc[:, i], group_B.iloc[:, i]])
                else:
                    raise ValueError('No provided mode for comparing.')
            result_compare.append((f_n, statistic_, pvalue_))
        res_df = pd.DataFrame(result_compare, columns=[
            'Feature', 'Statistic', 'p-value'])
        res_l = list(res_df.loc[res_df['p-value'] <= alpha, 'Feature'].values)
        if len(res_l) < 1:
            warnings.warn('No found DEGs')
        return res_l

    def run_semimatch(self,
                      semi=True,
                      nt_node='NT',
                      patch_meta=None,
                      return_all_gene=True,
                      return_nt_cell=True,
                      alpha=0.2,
                      n_neighbors=20,
                      **kwargs):
        # reference:https://github.com/theislab/pertpy/blob/main/pertpy/tools/_mixscape.py
        if len(self.node_dict) > 2:
            raise ValueError('The members are more than 2.')
        group_A, group_B = self.node_dict.values()
        group_A_exp = group_A.node_data['exp']
        group_B_exp = group_B.node_data['exp']
        if not group_A_exp.columns.equals(group_B_exp.columns):
            raise ValueError('The features should be consistent.')
        if semi:
            degs = self.run_distribute_test(alpha=alpha, mode='ad')
            if len(degs) < 1:
                f_sub = list(group_A_exp.columns.values)
            else:
                f_sub = degs
        else:
            f_sub = list(group_A_exp.columns.values)
        cell_A = list(group_A_exp.index.values)
        cell_B = list(group_B_exp.index.values)
        if group_A_exp is not None and group_B_exp is not None:
            merge_exp = pd.concat(
                [group_A_exp, group_B_exp], ignore_index=False)
        merge_exp_sub = merge_exp.loc[:, f_sub]

        mask_nt = pd.DataFrame(
            data=False, index=merge_exp.index, columns=['mask_nt'])
        nt_node = nt_node if isinstance(nt_node, list) else [nt_node]
        if group_A.node_name == nt_node:
            mask_nt.loc[cell_A, 'mask_nt'] = True
        elif group_B.node_name == nt_node:
            mask_nt.loc[cell_B, 'mask_nt'] = True
        else:
            raise ValueError(f'One node should be named by {nt_node}')
        if patch_meta is not None:
            cell_all = cell_A+cell_B
            mask_patch = patch_meta.loc[cell_all]
            if len(mask_patch) < len(mask_nt):
                raise ValueError("The cell_meta is inappropriate")
            mask_patch = mask_patch.rename('mask_patch')
            # mask_patch = mask_patch.reindex(merge_exp.index)
        else:
            mask_patch = pd.DataFrame(
                data='patch1', index=merge_exp.index, columns=['mask_patch'])
        mask_patch_wide = pd.get_dummies(
            mask_patch, columns=['mask_patch']).astype(bool)

        def _pca_select(mat, n_pcs=30):
            if min(mat.shape) >= n_pcs:
                pca = PCA(n_components=n_pcs,
                          svd_solver='auto', random_state=1)
                pca.fit(mat)
                cumulative_e_v = np.cumsum(
                    pca.explained_variance_ratio_)
                if max(cumulative_e_v) >= 0.95:
                    n_pc_95 = np.where(
                        cumulative_e_v >= 0.95)[0][0] + 1
                    pca_95 = PCA(n_components=n_pc_95, random_state=1)
                    return pca_95.fit_transform(mat)
                else:
                    return pca.transform(mat)
            else:
                if isinstance(mat, np.ndarray):
                    return mat
                elif isinstance(mat, pd.DataFrame):
                    return mat.to_numpy()
                else:
                    raise ValueError("Inappropiate format")

        if not return_all_gene:
            Sig_mat = merge_exp_sub.copy()
            NT_mat = merge_exp.loc[mask_nt['mask_nt'], f_sub]
        else:
            Sig_mat = merge_exp.copy()
            NT_mat = merge_exp.loc[mask_nt['mask_nt'], :]
        if not return_nt_cell:
            Sig_mat = Sig_mat.loc[~mask_nt['mask_nt'], :]

        for patch in mask_patch_wide.columns:
            bool_patch_all = mask_patch_wide[patch]
            bool_patch_nt = bool_patch_all & mask_nt['mask_nt']
            if not all(bool_patch_nt):
                bool_patch_nt = mask_nt['mask_nt']
            batch_nt_exp = merge_exp_sub.loc[bool_patch_nt, f_sub]
            batch_nt_2knn = _pca_select(batch_nt_exp)
            if return_nt_cell:
                # 返回nt+target所有的样本
                bool_patch_query = bool_patch_all
            else:
                # 只返回target样本，不返回nt样本
                bool_patch_query = bool_patch_all & ~mask_nt['mask_nt']
            if not any(bool_patch_query):
                continue
            batch_query_exp = merge_exp_sub.loc[bool_patch_query, f_sub]
            batch_query_2knn = _pca_select(batch_query_exp)
            eps = kwargs.pop("epsilon", 0.1)
            nn_index = NNDescent(
                batch_nt_2knn, n_neighbors=n_neighbors, **kwargs)
            indices, _ = nn_index.query(
                batch_query_2knn, k=n_neighbors, epsilon=eps)
            n_nt = batch_nt_2knn.shape[0]
            n_query = batch_query_2knn.shape[0]
            col_indices = np.ravel(indices)
            row_indices = np.repeat(np.arange(n_query), n_neighbors)
            neigh_matrix = csr_matrix(
                (np.ones_like(col_indices, dtype=np.float64),
                 (row_indices, col_indices)),
                shape=(n_query, n_nt),)
            neigh_matrix /= n_neighbors
            batch_nt_expm1 = np.expm1(NT_mat)
            mean_nt_per_query = np.log1p(neigh_matrix @ batch_nt_expm1)
            Sig_mat.loc[bool_patch_query, :] -= mean_nt_per_query
            return Sig_mat
