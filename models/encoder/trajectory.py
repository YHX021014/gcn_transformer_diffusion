from models.encoder.transformer import *
from models.encoder.gcn import *
from models.encoder.mlp import *


class TrajectoryModel(nn.Module):

    def __init__(self, args,
                 number_asymmetric_conv_layer=7, embedding_dims=32, number_gcn_layers=1, dropout=0, n_tcn=5,
                 num_heads=4, latent_dim=64, K=20, DEC_WITH_Z=True):
        super(TrajectoryModel, self).__init__()
        self.latent_dim = latent_dim
        self.K = K
        self.DEC_WITH_Z = DEC_WITH_Z
        self.args = args
        self.tem_encoder_obs = Transformer(embedding_dims, num_heads, dropout)
        self.tem_encoder_gt = Transformer(embedding_dims, num_heads, dropout)

        self.spa_encoder_obs = GCN(number_asymmetric_conv_layer, embedding_dims, number_gcn_layers,
                                   dropout, n_tcn, num_heads, 8)
        self.spa_encoder_gt = GCN(number_asymmetric_conv_layer, embedding_dims, number_gcn_layers,
                                  dropout, n_tcn, num_heads, 12)

        self.encoder_dest = MLP(input_dim=2, output_dim=embedding_dims * 2, hidden_size=(8, 16))
        self.fusion = MLP(input_dim=embedding_dims * 2, output_dim=embedding_dims, hidden_size=(8, 16))

        self.p_z_x = nn.Sequential(nn.Linear(embedding_dims * 8, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, latent_dim * 2))
        self.q_z_xy = nn.Sequential(nn.Linear(embedding_dims * 8 + embedding_dims * 12, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, latent_dim * 2))

        self.goal_decoder = nn.Sequential(nn.Linear(embedding_dims * 8 + latent_dim,
                                                    128),
                                          nn.ReLU(),
                                          nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 2))
        self.velocity_decoder = nn.Sequential(nn.Linear(embedding_dims * 8 + latent_dim,
                                                        128),
                                              nn.ReLU(),
                                              nn.Linear(128, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 24))
        self.enc_h_for_v = nn.Sequential(nn.Linear(self.K, 20),
                                         nn.ReLU())
        self.dec_init_hidden_size = embedding_dims * 8 + latent_dim if self.DEC_WITH_Z else embedding_dims * 8
        self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size, 256),
                                                nn.ReLU())
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(256, 256),
                                                    nn.ReLU())
        self.traj_dec_forward = nn.GRUCell(input_size=256,
                                           hidden_size=256)

        self.enc_h_to_back_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size, 256),
                                             nn.ReLU())

        self.traj_dec_input_backward = nn.Sequential(nn.Linear(2, 128),
                                                     nn.ReLU())

        self.velo_dec_input_backward = nn.Sequential(nn.Linear(2, 128),
                                                     nn.ReLU())

        self.traj_dec_backward = nn.GRUCell(input_size=256,
                                            hidden_size=256)

        self.traj_output = nn.Linear(256 * 2, 2)

    def cvae(self, obs_feature, gt_feature, target=None, z_mode=None, latent_dim=64):
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(obs_feature)
        z_mu_p = z_mu_logvar_p[:, :latent_dim]
        z_logvar_p = z_mu_logvar_p[:, latent_dim:]
        if target is not None:
            # 2. sample z from posterior, for training only
            z_mu_logvar_q = self.q_z_xy(torch.cat([obs_feature, gt_feature], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :latent_dim]
            z_logvar_q = z_mu_logvar_q[:, latent_dim:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp() / z_logvar_p.exp()) + \
                         (z_mu_p - z_mu_q).pow(2) / z_logvar_p.exp() - \
                         1 + \
                         (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0

        # 4. Draw sample
        K_samples = torch.randn(obs_feature.shape[0], self.K, latent_dim).cuda()
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.K, 1)

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD

    def pred_future_traj(self, dec_h, G, velocity):
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim)
            G: (Batch, K, pred_dim)
            velocity: (Batch, K, 24)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = 12

        K = G.shape[1]
        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len):  # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)

        forward_outputs = torch.stack(forward_outputs, dim=1)

        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        goal_input = self.traj_dec_input_backward(G)  # torch.cat([G])
        goal_input = goal_input.view(-1, goal_input.shape[-1])
        velocity_x = velocity[:, :, :12].unsqueeze(3)
        velocity_y = velocity[:, :, 12:].unsqueeze(3)
        v_input = torch.cat([velocity_x, velocity_y], dim=-1)
        velocity_input = self.velo_dec_input_backward(v_input[:, :, -1, :])
        velocity_input = velocity_input.view(-1, velocity_input.shape[-1])
        backward_input = torch.cat([goal_input, velocity_input], dim=1)
        for t in range(pred_len - 1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))  # (N*K, 2)
            goal_input = self.traj_dec_input_backward(output)  # (N*K, 128)
            velocity_input = self.velo_dec_input_backward(v_input[:, :, t, :])
            velocity_input = velocity_input.view(-1, velocity_input.shape[-1])
            backward_input = torch.cat([goal_input, velocity_input], dim=-1)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))

        # inverse because this is backward
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)

        return backward_outputs

    def cvae_loss(self, pred_goal, pred_traj, pred_velocity, target, velocity_x, velocity_y, best_of_many=True):
        '''
        CVAE loss use best-of-many
        Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
            best_of_many: whether use best of many loss or not
        Returns:

        '''
        K = pred_goal.shape[1]
        target = target.unsqueeze(2).repeat(1, 1, K, 1)
        pred_vel_x = pred_velocity[:, :, :12]
        pred_vel_y = pred_velocity[:, :, 12:]
        velocity_x = velocity_x.unsqueeze(1).repeat(1, K, 1)
        velocity_y = velocity_y.unsqueeze(1).repeat(1, K, 1)
        # print("%%%%", velocity_x.shape, pred_vel_x.shape)
        # select bom based on  goal_rmse
        goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :]) ** 2, dim=-1))
        traj_rmse = torch.sqrt(torch.sum((pred_traj - target) ** 2, dim=-1)).sum(dim=1)
        # print(traj_rmse.shape)
        velo_rmse = torch.sqrt(torch.sum((pred_vel_x - velocity_x) ** 2, dim=-1)) + \
                    torch.sqrt(torch.sum((pred_vel_y - velocity_y) ** 2, dim=-1))
        # print(velo_rmse.shape)
        if best_of_many:
            best_idx = torch.argmin(goal_rmse, dim=1)
            loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
            loss_velo = velo_rmse[range(len(best_idx)), best_idx].mean()
        else:
            loss_goal = goal_rmse.mean()
            loss_traj = traj_rmse.mean()
            loss_velo = velo_rmse.mean()

        return loss_goal, loss_traj, loss_velo

    def k_means(self, batch_x, ncluster=20, iter=10):
        """return clustering ncluster of x.

        Args:
            x (Tensor): B, K, 2
            ncluster (int, optional): Number of clusters. Defaults to 20.
            iter (int, optional): Number of iteration to get the centroids. Defaults to 10.
        """
        B, N, D = batch_x.size()
        batch_c = torch.Tensor().cuda()
        for i in range(B):
            x = batch_x[i]
            c = x[torch.randperm(N)[:ncluster]]
            for i in range(iter):
                a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
                c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
                nanix = torch.any(torch.isnan(c), dim=1)
                ndead = nanix.sum().item()
                c[nanix] = x[torch.randperm(N)[:ndead]]

            batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
        return batch_c

    def forward(self, graph_obs, identity_obs, obs_traj_rel, graph_gt=None, identity_gt=None, pred_traj_gt_rel=None, \
                velocity_x=None, velocity_y=None):

        # graph (1 obs_len N 3)    obs_traj (1,N, 2, obs_len)   des (1, N,2)  velocity (1,N,12)
        #############  obs_encoder  #################
        # get gcn feature
        obs_gcn_representation = self.spa_encoder_obs(graph_obs, identity_obs)  # (N, 8, 64)
        # get transformer feature
        self.obs_traj = obs_traj_rel.permute(0, 1, 3, 2)  # (1 N 8 2)
        obs_transformer_representation = self.tem_encoder_obs(self.obs_traj)
        # feature fusion
        obs_feature = torch.cat((obs_gcn_representation, obs_transformer_representation), dim=-1)
        obs_feature = self.fusion(obs_feature)  # (N,8,64)
        obs_feature = obs_feature.reshape(-1, obs_feature.shape[1] * obs_feature.shape[2])

        #############  gt_encoder  #################
        if graph_gt is not None:
            gt_gcn_representation = self.spa_encoder_gt(graph_gt, identity_gt)  # (N, 12, 64)
            # get transformer feature
            self.gt_traj = pred_traj_gt_rel.permute(0, 1, 3, 2)  # (1 N 12 2)
            gt_transformer_representation = self.tem_encoder_gt(self.gt_traj)
            # feature fusion
            gt_feature = torch.cat((gt_gcn_representation, gt_transformer_representation), dim=-1)
            gt_feature = self.fusion(gt_feature)  # (N,12,64)
            gt_feature = gt_feature.reshape(-1, gt_feature.shape[1] * gt_feature.shape[2])
        else:
            gt_feature = None
        Z, KLD = self.cvae(obs_feature, gt_feature, target=gt_feature, z_mode=None,
                           latent_dim=self.latent_dim)  # Z(N,20,64)
        enc_h_and_z = torch.cat([obs_feature.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        enc_h_and_z_forv = enc_h_and_z.permute(0, 2, 1)
        enc_h_and_z_forv = self.enc_h_for_v(enc_h_and_z_forv)
        enc_h_and_z_forv = enc_h_and_z_forv.permute(0, 2, 1)

        pred_goal = self.goal_decoder(enc_h_and_z)  # (N,20,2)
        pred_velocity = self.velocity_decoder(enc_h_and_z_forv)  # (N, 20, 24)
        fine_goal = self.k_means(pred_goal)
        # print("$$$$$$",fine_goal.shape, pred_velocity.shape)
        dec_h = enc_h_and_z_forv if self.DEC_WITH_Z else obs_feature
        pred_traj = self.pred_future_traj(dec_h, fine_goal, pred_velocity)  # (N,12,20,2)
        # 5. compute loss

        if graph_gt is not None:
            # train and val
            gt_traj = self.gt_traj
            loss_goal, loss_traj, loss_velo = self.cvae_loss(fine_goal,
                                                             pred_traj,
                                                             pred_velocity,
                                                             gt_traj.squeeze(),
                                                             velocity_x.squeeze(),
                                                             velocity_y.squeeze(),
                                                             best_of_many=True
                                                             )
            loss_dict = {'loss_goal': loss_goal, 'loss_traj': loss_traj, 'loss_kld': KLD, 'loss_velo': loss_velo}
        else:
            # test
            loss_dict = {}

        return pred_goal, pred_traj, loss_dict