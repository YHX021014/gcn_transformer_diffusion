import os
import numpy as np
import torch

def print_info(epoch, model, optimizer, loss_dict, logger):


    info = "Epoch:{},\t lr:{:6},\t loss_goal:{:.4f},\t loss_traj:{:.4f},\t loss_kld:{:.4f},\t \
            loss_velo:{:.4f} ".format(
            epoch, optimizer.param_groups[0]['lr'], loss_dict['loss_goal'], loss_dict['loss_traj'],
            loss_dict['loss_kld'], loss_dict['loss_velo'])
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])

    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)

def viz_results(viz,
                X_global,
                y_global,
                pred_traj,
                img_path,
                dist_goal,
                dist_traj,
                bbox_type='cxcywh',
                normalized=True,
                logger=None,
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    id_to_show = np.random.randint(pred_traj.shape[0])

    # 1. initialize visualizer
    viz.initialize()

    # 2. visualize point trajectory or box trajectory
    if y_global.shape[-1] == 2:
        viz.visualize(pred_traj[id_to_show], color=(0, 1, 0), label='pred future', viz_type='point')
        viz.visualize(X_global[id_to_show], color=(0, 0, 1), label='past', viz_type='point')
        viz.visualize(y_global[id_to_show], color=(1, 0, 0), label='gt future', viz_type='point')
    elif y_global.shape[-1] == 4:
        T = X_global.shape[1]
        viz.visualize(pred_traj[id_to_show], color=(0, 255., 0), label='pred future', viz_type='bbox',
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])
        viz.visualize(X_global[id_to_show], color=(0, 0, 255.), label='past', viz_type='bbox',
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(y_global[id_to_show], color=(255., 0, 0), label='gt future', viz_type='bbox',
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])

    # 3. optinaly visualize GMM distribution
    if hasattr(dist_goal, 'mus') and viz.mode == 'plot':
        dist = {'mus':dist_goal.mus.numpy(), 'log_pis':dist_goal.log_pis.numpy(), 'cov': dist_goal.cov.numpy()}
        viz.visualize(dist, id_to_show=id_to_show, viz_type='distribution')

    # 4. get image.
    if y_global.shape[-1] == 2:
        viz_img = viz.plot_to_image(clear=True)
    # else:
    #     viz_img = viz.img

        if hasattr(logger, 'log_image'):
            logger.log_image(viz_img, label=name)

