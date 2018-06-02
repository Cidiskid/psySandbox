import arg
from env import Env, State, Area, get_area_sample_distr
from agent import Agent
from copy import deepcopy
import acts
from util import moniter
from util.config import all_config
import logging


class Control:
    def __init__(self):
        self.global_arg = arg.init_global_arg()
        self.main_env = Env(arg.init_env_arg(self.global_arg))
        self.agents = []

    def run_frame(self, Ti, Tfi, up_info):
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

        for i in range(len(self.agents)):
            if (not 'max' in self.agents[i].inter_area.info):
                self.agents[i].inter_area.info = get_area_sample_distr(env=self.main_env, T=0,
                                                                       area=self.agents[i].inter_area,
                                                                       state=self.agents[i].state_now,
                                                                       sample_num=
                                                                       self.agents[i].frame_arg['ACT']['hqxx'][
                                                                           'sample_n'],
                                                                       dfs_r=self.agents[i].frame_arg['ACT']['hqxx'][
                                                                           'dfs_p'])

            if (self.agents[i].frame_arg['PROC']['action']):
                if (self.main_env.getValue(self.agents[i].state_now, Ti) >= self.agents[i].inter_area.info['max']*0.99):
                    logging.debug("Agent %d, act_hqxx" % (i))
                    self.agents[i] = acts.act_hqxx(self.main_env, self.agents[i], Ti, Tfi)
                else:
                    logging.debug("Agent %d, act_xdzx" % (i))
                    self.agents[i] = acts.act_xdzx(self.main_env, self.agents[i], Ti, Tfi)
            else:
                pass

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

    def run_exp(self):
        up_info = {}
        for i in range(self.global_arg["Nagent"]):
            self.agents.append(Agent(arg.init_agent_arg(self.global_arg,
                                                        self.main_env.arg)))
            self.agents[i].state_now = State([0 for _ in range(self.main_env.N)])

        stage_num = self.global_arg['T'] // self.global_arg['Ts']
        for k in range(self.global_arg["Nagent"]):
            csv_head = ['frame', 'SSMfi', 'SSM_f-req', 'proc_action', 'SSM_f_need',
                        'nkmax', 'nkmin', 'nkmid', 'nkavg', 'nk0.75', "nk0.25"]
            #                        'peakmax', 'peakmin', 'peakmid', 'peakavg', 'peak0.75', "peak0.25"]
            moniter.AppendToCsv(csv_head, all_config['result_csv_path'][k])
        for i in range(stage_num):
            Ti = i * self.global_arg['Ts'] + 1
            logging.info("stage %3d , Ti:%3d" % (i, Ti))
            up_info['nkinfo'] = self.main_env.getModelDistri(Ti)
            #            up_info['nk_peak'] = self.main_env.getModelPeakDistri(Ti)
            self.run_stage(Ti, up_info)


if (__name__ == "__main__"):
    import time
    import os

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
    main_control = Control()
    main_control.run_exp()
