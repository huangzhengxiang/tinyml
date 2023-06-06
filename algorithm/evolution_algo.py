import random
import heapq
import threading
from copy import deepcopy

from easydict import EasyDict
from train_cls import main, build_config

MAX_BIAS_UPDATE = 20  # 最多更新20层bias
MAX_WEIGHT_UPDATE = 5   # 最多更新5层weight
RATIO = [0.125, 0.25, 0.5, 1.0]
INDIVIDUAL_ONE_GENERATION = 5  # 每一代个体数量
PARENTS = 2  # 每一代剩下的亲本数量
MUTATION_PROB = 0.25  # 每一个参数变异的概率
THRESHOLD = 85  # 结束遗传的准确率阈值
individual_val = []  # 记录个体适应性（即训练准确率）
individual_param = []  # 记录个体信息

'''
class MyThread(threading.Thread):
    def run(self):
        n_bias_i, weight_idx_str_i, weight_ratio_str_i = generate_individual()
        new_configs = get_evolution_config(n_bias_i, weight_idx_str_i, weight_ratio_str_i)
        build_config(new_configs)
        val_info_dict = main()
        return val_info_dict['val/best']
'''

def is_mutation():
    return random.random() < MUTATION_PROB


def random_walk():
    rand = random.random()
    if rand > 0.8:
        return 2
    elif rand > 0.5:
        return 1
    elif rand > 0.2:
        return -1
    else:
        return -2


def mutation(input_param):
    # n_bias变异
    param = deepcopy(input_param)
    if is_mutation():
        param['n_bias'] = min(max(1, param['n_bias'] + random_walk()), MAX_BIAS_UPDATE)
    # weight_idx变异
    for [idx, layer] in enumerate(param['weight_idx']):
        if is_mutation():
            new_layer = min(max(42 - param['n_bias'], layer + random_walk()), 41)
            if new_layer not in param['weight_idx']:
                param['weight_idx'][idx] = new_layer
        if is_mutation():  # 随机分配新的ratio
            param['weight_ratio'][idx] = RATIO[random.randint(0, 3)]
    param['weight_idx'] = sorted(param['weight_idx'])
    print('new parameter:', param)
    return param


def generate_individual():
    n_bias = random.randint(1, MAX_BIAS_UPDATE)
    n_weights = random.randint(1, min(MAX_WEIGHT_UPDATE, n_bias))  # weight层数不超过bias层数
    weight_idx = sorted(random.sample(range(42 - n_bias, 42), n_weights))
    weight_ratio = [RATIO[random.randint(0, len(RATIO) - 1)] for _ in range(n_weights)]

    return {'n_bias': n_bias, 'weight_idx': weight_idx, 'weight_ratio': weight_ratio}


def get_evolution_config(individual_param):
    # 添加训练时的默认配置
    configs = EasyDict()
    configs.run_dir = 'runs/flowers/mcunet-5fps/sparse_100kb/sgd_qas_nomom'
    configs.manual_seed = 0
    configs.evaluate = False
    configs.ray_tune = 0
    configs.resume = 0
    configs.data_provider = {'dataset': 'image_folder',
                             'root': '~/dataset/flowers102',
                             'resize_scale': 0.08,
                             'color_aug': 0.4,
                             'base_batch_size': 64,
                             'n_worker': 8,
                             'image_size': 128,
                             'num_classes': 102}
    configs.run_config = {'n_epochs': 5,  # 修改epoch，最初为50
                          'base_lr': 0.025,
                          'bs256_lr': 0.1,
                          'warmup_epochs': 1,  # 修改epoch，最初为5
                          'warmup_lr': 0,
                          'lr_schedule_name': 'cosine',
                          'weight_decay': 0,
                          'no_wd_keys': ['norm', 'bias'],
                          'optimizer_name': 'sgd_scale_nomom',
                          'bias_only': 0,
                          'fc_only': 0,
                          'fc_lr10': 0,
                          'eval_per_epochs': 10,  # 修改测试间隔，最初为10
                          'grid_output': None,
                          'grid_ckpt_path': None,
                          'n_block_update': -1}
    configs.net_config = {'net_name': 'mcunet-5fps',
                          'pretrained': False,
                          'cls_head': 'linear',
                          'dropout': 0.0,
                          'mcu_head_type': 'fp'}
    weight_idx = individual_param['weight_idx']
    weight_ratio = individual_param['weight_ratio']
    weight_idx_str = str(weight_idx[0])
    weight_ratio_str = str(weight_ratio[0])
    for i in range(1, len(weight_idx)):
        weight_idx_str += ('-' + str(weight_idx[i]))
        weight_ratio_str += ('-' + str(weight_ratio[i]))
    print(individual_param['n_bias'], weight_idx_str, weight_ratio_str)
    configs.backward_config = {'enable_backward_config': 1, 'n_bias_update': individual_param['n_bias'], 'n_weight_update': None,
                               'weight_update_ratio': weight_ratio_str, 'weight_select_criteria': 'magnitude+',
                               'pw1_weight_only': 0, 'manual_weight_idx': weight_idx_str, 'quantize_gradient': 0,
                               'freeze_fc': 0, 'train_scale': 0}
    return configs


if __name__ == '__main__':
    # 生成初代个体并训练
    # 随机选择5组参数：（n_bias_update, manual_weight_idx, weight_update_ratio），每组训练10个epoch，选出效果最好的两组交叉重组，然后变异再产生5组，如此循环下去
    r = 1
    result = []
    individual_param.clear()
    individual_val.clear()
    for idx in range(INDIVIDUAL_ONE_GENERATION):
        param = generate_individual()
        individual_param.append(param)
        new_configs = get_evolution_config(param)
        build_config(new_configs)
        val_info_dict = main()
        individual_val.append(val_info_dict['val/best'])
    max_individual_idx = list(map(individual_val.index, heapq.nlargest(PARENTS, individual_val)))
    print('Current best val: ', individual_val[max_individual_idx[0]], '  Current best param: ',
          individual_param[max_individual_idx[0]])
    result.append([r, individual_val[max_individual_idx[0]], individual_param[max_individual_idx[0]]])

    while individual_val[max_individual_idx[0]] < THRESHOLD:
        # 亲代变异得到子代
        r += 1
        for child_idx in range(INDIVIDUAL_ONE_GENERATION):
            select_parent = 0  # 可以用一个权重函数选择各亲本的子代比例，此处无脑选择最强的
            new_param = mutation(individual_param[max_individual_idx[select_parent]])
            individual_param.append(new_param)
            new_configs = get_evolution_config(new_param)
            build_config(new_configs)
            val_info_dict = main()
            individual_val.append(val_info_dict['val/best'])

        max_individual_idx = list(map(individual_val.index, heapq.nlargest(PARENTS, individual_val)))
        print('Current best val: ', individual_val[max_individual_idx[0]], '  Current best param: ',
              individual_param[max_individual_idx[0]])
        result.append([r, individual_val[max_individual_idx[0]], individual_param[max_individual_idx[0]]])
        if r % 5 == 0:
            print('Show result: ', result)

