# -*- coding:utf-8 -*-

import arg
from env import Env, State, Area, get_area_sample_distr
from agent import Agent
from copy import deepcopy
import acts
from util import moniter
from util.config import all_config
import logging
from brain import using_brain

class Control:
    def __init__(self):
        self.global_arg = arg.init_global_arg()
        self.main_env = Env(arg.init_env_arg(self.global_arg))
        self.agents = []

    # Frame运行过程，Ti表示所处stage的第一帧时间戳，Tfi表示stage中第几帧的时间戳
    def run_frame(self, Ti, Tfi, up_info):
        # 将每个Agent的中间状态初始化
        for i in range(len(self.agents)):
            last_arg = deepcopy(self.agents[i].frame_arg)
            #            logging.debug("agent %d, %s"%(i,"{}".format(self.agents[i].frame_arg)))
            self.agents[i].frame_arg = arg.init_frame_arg(
                global_arg=self.global_arg,
                env_arg=self.main_env.arg,
                agent_arg=self.agents[i].agent_arg,
                stage_arg=self.agents[i].stage_arg,
                last_arg=last_arg,
                Tp=Ti,
                PSMfi=self.main_env.getValue(self.agents[i].state_now, Ti)
            )

        # 每个Agent依次按Brain中的策略行动
        for i in range(len(self.agents)):
            self.agents[i] = using_brain[i](self.main_env,
                                            self.agents[i],
                                              Ti, Tfi, i)
    # Stage运行过程
    def run_stage(self, Ti, up_info):
        for i in range(len(self.agents)):
            last_arg = deepcopy(self.agents[i].stage_arg)
            self.agents[i].stage_arg = arg.init_stage_arg(self.global_arg,
                                                          self.main_env.arg,
                                                          self.agents[i].agent_arg,
                                                          last_arg,
                                                          Ti)
        for i in range(self.global_arg['Ts']):
            logging.info("frame %3d , Ti:%3d" % (i, Ti))

            # 运行Frame
            self.run_frame(Ti, i, up_info)
            for k in range(self.global_arg["Nagent"]):
                csv_info = [
                    Ti + i,
                    self.main_env.getValue(self.agents[k].state_now, Ti),
                    self.agents[k].frame_arg['PSM']['f-req'],
                    int(self.agents[k].frame_arg['PROC']['action']),
                    self.agents[k].frame_arg['PSM']['a-need'],
                    up_info['nkinfo']['max'],
                    up_info['nkinfo']['min'],
                    up_info['nkinfo']['mid'],
                    up_info['nkinfo']['avg'],
                    up_info['nkinfo']['p0.75'],
                    up_info['nkinfo']['p0.25']
                ]
                moniter.AppendToCsv(csv_info, all_config['result_csv_path'][k])
            agent_value = [self.main_env.getValue(self.agents[k].state_now, Ti) for k in
                           range(self.global_arg["Nagent"])]
            csv_info = [Ti + i] \
                       + agent_value \
                       + [sum(agent_value) / len(agent_value)] \
                       + [up_info['nkinfo'][key] for key in ['max', 'min', 'mid', 'avg', 'p0.75', 'p0.25']]
            moniter.AppendToCsv(csv_info, all_config['result_csv_path'][-1])

# 实验运行过程
    def run_exp(self):
        up_info = {}

        for i in range(self.global_arg["Nagent"]):
            self.agents.append(Agent(arg.init_agent_arg(self.global_arg,
                                                        self.main_env.arg)))
            self.agents[i].state_now = State([0 for _ in range(self.main_env.N)])
            self.agents[i].inter_area.info = get_area_sample_distr(env=self.main_env, T=0,
                                                                   area=self.agents[i].inter_area,
                                                                   state=self.agents[i].state_now,
                                                                   sample_num=self.main_env.arg['ACT']['hqxx'][
                                                                       'sample_n'],
                                                                   dfs_r=self.main_env.arg['ACT']['hqxx']['dfs_p'])

        stage_num = self.global_arg['T'] // self.global_arg['Ts']
        for k in range(self.global_arg["Nagent"]):
            csv_head = ['frame', 'SSMfi', 'SSM_f-req', 'proc_action', 'SSM_f_need',
                        'nkmax', 'nkmin', 'nkmid', 'nkavg', 'nk0.75', "nk0.25"]
            #                        'peakmax', 'peakmin', 'peakmid', 'peakavg', 'peak0.75', "peak0.25"]
            moniter.AppendToCsv(csv_head, all_config['result_csv_path'][k])
        csv_head = ['frame'] \
                   + ["%s%d" % (using_brain[k].func_name,k) for k in range(self.global_arg['Nagent'])] \
                   + ["agent_avg"] \
                   + ['nkmax', 'nkmin', 'nkmid', 'nkavg', 'nk0.75', "nk0.25"]
        moniter.AppendToCsv(csv_head, all_config['result_csv_path'][-1])
        for i in range(stage_num):
            Ti = i * self.global_arg['Ts'] + 1
            logging.info("stage %3d , Ti:%3d" % (i, Ti))
            up_info['nkinfo'] = self.main_env.getModelDistri(Ti)
            #            up_info['nk_peak'] = self.main_env.getModelPeakDistri(Ti)

            # 运行一个Stage，Ti表示每个Stage的第一帧
            self.run_stage(Ti, up_info)


if (__name__ == "__main__"):
    import time
    import os
# 准备工作，初始化实验环境，生成实验结果文件夹等
    all_config.load()
    moniter.LogInit()
    logging.info("Start")
    global_arg = arg.init_global_arg()
    env_arg = arg.init_env_arg(global_arg)
    exp_id = "_".join([
        "sigleview",
        time.strftime("%Y%m%d-%H%M%S"),
        "N" + str(env_arg['N']),
        "K" + str(env_arg['K']),
        "P" + str(env_arg['P']),
        "T" + str(global_arg['T']),
        "Ts" + str(global_arg['Ts'])
    ])
    try:
        os.mkdir(os.path.join("result", exp_id))
    except:
        pass
    all_config['result_csv_path'] = [
        os.path.join("result", exp_id, "res_%s_%02d.csv" % (exp_id, i)) for i in range(global_arg["Nagent"])
    ]
    all_config['result_csv_path'].append(
        os.path.join("result", exp_id, "res_%s_overview.csv" % (exp_id))
    )
    main_control = Control()
    main_control.run_exp()  # 开始运行实验
