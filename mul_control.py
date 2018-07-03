# -*- coding:utf-8 -*-
import logging
import arg
from group import Group, SoclNet
from env import Env, Area, State, get_area_sample_distr
from agent import Agent
from copy import deepcopy
import brain
import meeting
from util.config import all_config
from util import moniter
from record import Record
from random import randint


class MulControl:
    def __init__(self):
        # 环境初始化
        self.global_arg = arg.init_global_arg()
        env_arg = arg.init_env_arg(self.global_arg)
        # 增加nk的一个读入操作
        self.main_env = Env(env_arg)
        if all_config['checkpoint']['env']['enable']:
            self.main_env.nkmodel_load(all_config['checkpoint']['env']['path'])
        self.main_env.nkmodel_save(all_config["nkmodel_path"])
        # 个体初始化
        self.agents = []
        csv_head_agent = ['agent_no'] + ['st_state'] + ['insight'] + ['xplr'] + ['xplt'] + ['enable']
        moniter.AppendToCsv(csv_head_agent, all_config['agent_csv_path'])
        for i in range(self.global_arg["Nagent"]):
            # 个体随机初始位置
            start_st_label = [randint(0, self.main_env.P - 1) for j in range(self.main_env.N)]
            state_start = State(start_st_label)
            self.agents.append(Agent(arg.init_agent_arg(self.global_arg,
                                                        self.main_env.arg),
                                     self.main_env))
            self.agents[i].state_now = deepcopy(state_start)

            # TODO cid 去除了一开始给一个全局area，改为添加一个包含起点的点area
            start_area = Area(self.agents[i].state_now, [False] * self.main_env.N, 0)
            start_area.info = get_area_sample_distr(env=self.main_env, area=start_area, state=self.agents[i].state_now,
                                                    T_stmp=0, sample_num=1, dfs_r=1)
            self.agents[i].renew_m_info(start_area, 0)
            self.a_plan = None
            logging.info("state:%s, insight:%.5s ,xplr:%.5s, xplt:%.5s, enable:%.5s" % (
                str(self.agents[i].state_now),
                self.agents[i].agent_arg['a']['insight'],
                self.agents[i].agent_arg['a']['xplr'],
                self.agents[i].agent_arg['a']['xplt'],
                self.agents[i].agent_arg['a']['enable']))
            # 记录agent信息
            csv_info_agent = ['agent%d' % i] \
                             + [self.agents[i].state_now] \
                             + [self.agents[i].agent_arg['a']['insight']] \
                             + [self.agents[i].agent_arg['a']['xplr']] \
                             + [self.agents[i].agent_arg['a']['xplt']] \
                             + [self.agents[i].agent_arg['a']['enable']]
            moniter.AppendToCsv(csv_info_agent, all_config['agent_csv_path'])

        # 社会网络初始化
        soclnet_arg = arg.init_soclnet_arg(self.global_arg, env_arg)
        self.socl_net = SoclNet(soclnet_arg)
        self.socl_net.new_flat_init() # 修改初始化方法
        # self.socl_net.flat_init()
        if all_config['checkpoint']['socl_network']['enable']:
            self.socl_net.power_load(all_config['checkpoint']['socl_network']['power'])
            self.socl_net.relat_load(all_config['checkpoint']['socl_network']['relat'])
        self.record = Record()

    def run_meet_frame(self, Ti, Tfi, meet_name, member, host, up_info):
        # 根据m_name开会
        logging.debug("m_name:%s, member:%s, host:%s" % (meet_name, member, host))
        self.agents, self.socl_net = meeting.meet_map[meet_name](env=self.main_env,
                                                                 agents=self.agents,
                                                                 member=member,
                                                                 host=host,
                                                                 socl_net=self.socl_net,
                                                                 record=self.record,
                                                                 T=Ti, Tfi=Tfi)

    def run_all_frame(self, Ti, Tfi, meet_req, up_info):
        # 将每个Agent上一帧的初始拷贝进来
        for i in range(len(self.agents)):
            last_arg = deepcopy(self.agents[i].frame_arg)
            # logging.debug("agent %d, %s"%(i,"{}".format(self.agents[i].frame_arg)))
            self.agents[i].frame_arg = arg.init_frame_arg(
                global_arg=self.global_arg,
                env_arg=self.main_env.arg,
                agent_arg=self.agents[i].agent_arg,
                stage_arg=self.agents[i].stage_arg,
                last_arg=last_arg,
                Tp=Ti,
                PSMfi=self.main_env.getValue(self.agents[i].state_now, Ti)
            )
        logging.debug("agent copy finished")
        # 清空agent的行动和会议记录
        for i in range(len(self.agents)):
            self.agents[i].meeting_now = ''
            self.agents[i].policy_now = ''

        # NOTE cid 增加SoclNet自然衰减
        self.socl_net.relat_cd(self.socl_net.arg['re_decr_r'])

        # 读取之前发起的集体行动
        all_host = set()
        all_meet_info = {}
        new_meet_req = {}
        # 把每一种meeting的host先集中起来，并加入到对应的meet_info中
        # meet_req的结构大致如下？
        # meet_req={
        #    "m_name1":{agent}
        #    "m_name2":{agent}
        # }
        # m_name是指信息交流xxjl之类的集体行动名称

        for m_name in meet_req:
            all_host = all_host.union(meet_req[m_name])
            all_meet_info[m_name] = {"member": deepcopy(meet_req[m_name]),
                                     "host": deepcopy(meet_req[m_name])}
        # 询问每个Agent是否加入
        logging.debug("all host:%s" % (all_host))
        for m_name in all_meet_info:
            logging.debug("before m_name:%s, member:%s, host:%s" % (
                m_name, all_meet_info[m_name]['member'], all_meet_info[m_name]['host']))
        for i in range(len(self.agents)):
            #            logging.debug("all_host:{}".format(all_host))
            # 跳过所有host
            if i in all_host:
                continue
            # 返回是否参与集体行动的信息，如果不参与，执行完个体行动，如果参与,进入后续run_meet_frame
            if self.global_arg['mul_agent']:
                # logging.info("using mul_act")
                self.agents[i], self.socl_net, meet_info = brain.mul_agent_act(env=self.main_env,
                                                                               soc_net=self.socl_net,
                                                                               agent=self.agents[i],
                                                                               Ti=Ti, Tfi=Tfi, agent_no=i,
                                                                               record=self.record,
                                                                               meet_req=meet_req)
            else:
                self.agents[i], self.socl_net, meet_info = brain.sgl_agent_act(env=self.main_env,
                                                                               soc_net=self.socl_net,
                                                                               agent=self.agents[i],
                                                                               Ti=Ti, Tfi=Tfi, agent_no=i,
                                                                               record=self.record,
                                                                               meet_req=meet_req)

            if meet_info is None:
                continue
            # 选择参加会议，则加入会议名单
            if meet_info['type'] == 'commit':
                all_meet_info[meet_info['name']]["member"].add(i)
            # 选择发起新会议
            if meet_info['type'] == 'req':
                if not meet_info['name'] in new_meet_req:
                    new_meet_req[meet_info['name']] = set()
                new_meet_req[meet_info['name']].add(i)
        # 每个host都选完人之后，依次开会
        for m_name in all_meet_info:
            logging.debug("after m_name:%s, member:%s, host:%s" % (
                m_name, all_meet_info[m_name]['member'], all_meet_info[m_name]['host']))
            self.run_meet_frame(Ti, Tfi, m_name,
                                all_meet_info[m_name]['member'],
                                all_meet_info[m_name]['host'],
                                up_info)
        return new_meet_req

    def run_stage(self, Ti, meet_req, up_info):
        # 将Agent上一个stage的最终状态拷贝过来
        for i in range(len(self.agents)):
            last_arg = deepcopy(self.agents[i].stage_arg)
            self.agents[i].stage_arg = arg.init_stage_arg(self.global_arg,
                                                          self.main_env.arg,
                                                          self.agents[i].agent_arg,
                                                          last_arg,
                                                          Ti)
        meet_req = {}
        #  NOTE cid传了个up_info进去，避免重复遍历
        self.record.add_env_record(self.main_env, Ti, up_info)
        self.record.add_socl_net_record(self.socl_net, Ti)
        for i in range(self.global_arg['Ts']):
            logging.info("frame %3d , Ti:%3d" % (i, Ti))
            self.record.add_agents_record(self.main_env, self.agents, Ti + i)
            # 运行Frame， 并将运行后生成的会议请求记录下来
            meet_req = self.run_all_frame(Ti, i, meet_req, up_info)

            # 输出每个个体的具体信息
            for k in range(self.global_arg["Nagent"]):
                tmp_goal = ''
                tmp_goal_value = ''
                if not self.agents[k].a_plan is None:
                    tmp_goal = self.agents[k].a_plan.goal
                    tmp_goal_value = self.agents[k].a_plan.goal_value

                csv_info_result = [
                    Ti + i,
                    str(self.agents[k].state_now),
                    self.main_env.getValue(self.agents[k].state_now, Ti),
                    self.agents[k].get_max_area().info['max'],
                    str(self.agents[k].get_max_area().center),
                    str(self.agents[k].policy_now) + '&' + str(self.agents[k].meeting_now),
                    str(tmp_goal),
                    tmp_goal_value
                ]
                moniter.AppendToCsv(csv_info_result, all_config['result_csv_path'][k])

            # 输出当前value
            agent_value = [self.main_env.getValue(self.agents[k].state_now, Ti) for k in
                           range(self.global_arg["Nagent"])]
            agent_avg = sum(agent_value) / len(agent_value)

            csv_info_value = [Ti + i] \
                             + agent_value \
                             + [agent_avg, max(agent_value), min(agent_value)] \
                             + [up_info['nkinfo'][key] for key in ['max', 'min', 'avg']] \
                             + [(agent_avg - up_info['nkinfo']['min']) / (
                    up_info['nkinfo']['max'] - up_info['nkinfo']['min'])]
            moniter.AppendToCsv(csv_info_value, all_config['value_csv_path'][-1])

            # 输出max_area
            agent_max_area = [self.agents[k].get_max_area().info['max'] for k in
                              range(self.global_arg["Nagent"])]
            csv_info_area = [Ti + i] \
                            + agent_max_area \
                            + [sum(agent_max_area) / len(agent_max_area)] \
                            + [up_info['nkinfo']['max']]
            moniter.AppendToCsv(csv_info_area, all_config['area_csv_path'])

            # NOTE cid 添加act信息(相应增加agent类里的变量）
            act_list = [self.agents[k].policy_now + '&' + self.agents[k].meeting_now for k in
                        range(self.global_arg["Nagent"])]
            csv_info_act = [Ti + i] \
                           + act_list
            moniter.AppendToCsv(csv_info_act, all_config['act_csv_path'])

            # TODO @wzk 按stage输出
        if self.global_arg['mul_agent']:
            # net_title, net_data = self.record.output_socl_net_per_frame(Ti + i)
            power_save_path = os.path.join(all_config['network_csv_path'], "power_%04d.csv" % (Ti))
            relat_save_path = os.path.join(all_config['network_csv_path'], "relat_%04d.csv" % (Ti))
            self.socl_net.power_save(power_save_path)
            self.socl_net.relat_save(relat_save_path)
            #  P1-05 增加Socil Network的结果输出
        return meet_req

    def run_exp(self):
        up_info = {}

        # 单个agent的结果表
        for k in range(self.global_arg["Nagent"]):
            csv_head = ['frame', 'state', 'value', 'area_v', 'area_center', 'act', 'goal', 'goal_value']
            moniter.AppendToCsv(csv_head, all_config['result_csv_path'][k])
        # 结果汇总表
        # 添加agent max和agent min
        csv_head_value = ['frame'] \
                         + ["agent%d" % (k) for k in range(self.global_arg['Nagent'])] \
                         + ["agent_avg", "agent_max", "agent_min"] \
                         + ['peakmax', 'peakmin', 'peakavg'] \
                         + ['adj_avg']
        moniter.AppendToCsv(csv_head_value, all_config['value_csv_path'][-1])
        csv_head_area = ['frame'] \
                        + ["agent%d" % (k) for k in range(self.global_arg['Nagent'])] \
                        + ["agent_avg"] \
                        + ['nkmax']
        moniter.AppendToCsv(csv_head_area, all_config['area_csv_path'])

        csv_head_act = ['frame'] \
                       + ["agent%d" % (k) for k in range(self.global_arg['Nagent'])]
        moniter.AppendToCsv(csv_head_act, all_config['act_csv_path'])

        stage_num = self.global_arg['T'] // self.global_arg['Ts']

        # self.main_env.getModelDistri() # 为了作图，仅测试时调用！！
        up_info['nkinfo'] = self.main_env.getModelPeakDistri()  # 将nkinfo变为peakvalue
        # all_peak_value = self.main_env.getAllPeakValue()
        # moniter.DrawHist(all_peak_value, all_config['peak_hist'])

        meet_req = {}
        for i in range(stage_num):
            Ti = i * self.global_arg['Ts'] + 1
            logging.info("stage %3d, Ti:%3d" % (i, Ti))
            self.main_env.T_clock = Ti
            # 每个stage遍历一遍当前模型，获取分布信息
            # 减少运算量，只算第一帧
            # up_info['nkinfo'] = self.main_env.getModelDistri()
            # logging.debug("max_value:{max}".format(**up_info['nkinfo']))
            # 运行一个Stage，Ti表示每个Stage的第一帧
            meet_req = self.run_stage(Ti, meet_req, up_info)


if __name__ == '__main__':
    import time
    import os

    # 准备工作，初始化实验环境，生成实验结果文件夹等
    all_config.load()
    moniter.LogInit()
    logging.info("Start")
    global_arg = arg.init_global_arg()
    env_arg = arg.init_env_arg(global_arg)
    # 总目录
    batch_id = '_'.join([
        'batch',
        time.strftime("%Y%m%d-%H%M%S"),
        "N" + str(env_arg['N']),
        "K" + str(env_arg['K']),
        "P" + str(env_arg['P']),
        "T" + str(global_arg['T']),
        "Ts" + str(global_arg['Ts'])
    ])
    all_config['batch_id'] = batch_id
    all_config['exp_list_path'] = os.path.join("result", all_config['batch_id'], 'exp_list.csv')
    exp_list = []
    try:
        os.mkdir(os.path.join("result", all_config['batch_id']))
    except:
        pass

    # 运行实验主体
    for exp_num in range(global_arg['repeat']):
        if global_arg['mul_agent']:
            exp_id = "_".join([
                "mul",
                time.strftime("%Y%m%d-%H%M%S"),
                "exp" + str(exp_num)
            ])
        else:
            exp_id = "_".join([
                "sgl",
                time.strftime("%Y%m%d-%H%M%S"),
                "exp" + str(exp_num)
            ])
        all_config['exp_id'] = exp_id
        exp_list.append(exp_id)

        try:
            os.mkdir(os.path.join("result", all_config['batch_id'], exp_id))
            os.mkdir(os.path.join("result", all_config['batch_id'], exp_id, 'detail'))
        except:
            pass
        # 单个结果文档输出,位于detail目录下
        all_config['result_csv_path'] = [
            os.path.join("result", all_config['batch_id'], exp_id, "detail", "%s_%02d.csv" % (exp_id, i)) for i in
            range(global_arg["Nagent"])
        ]
        # network输出，位于network下
        if global_arg['mul_agent']:
            all_config['network_csv_path'] = os.path.join("result", all_config['batch_id'], exp_id, "network")
            try:
                os.mkdir(all_config['network_csv_path'])
            except:
                pass
        # 其他汇总信息，位于每个exp的总目录下
        all_config['value_csv_path'] = []
        all_config['value_csv_path'].append(
            os.path.join("result", all_config['batch_id'], exp_id, "%s_value.csv" % (exp_id))
        )

        # max_area输出
        all_config['area_csv_path'] = os.path.join("result", all_config['batch_id'], exp_id, "%s_area.csv" % (exp_id))
        # 添加一个act的记录文件
        all_config['act_csv_path'] = os.path.join("result", all_config['batch_id'], exp_id, "%s_act.csv" % (exp_id))

        all_config['nkmodel_path'] = os.path.join("result", all_config['batch_id'], exp_id,
                                                  "%s_nkmodel.pickle" % (exp_id))
        all_config['agent_csv_path'] = os.path.join("result", all_config['batch_id'], exp_id,
                                                    "%s_agent_attribute.csv" % (exp_id))
        all_config['peak_hist'] = os.path.join("result", all_config['batch_id'], exp_id, "%s_peak_hist.png" % (exp_id))
        all_config['total_hist'] = os.path.join("result", all_config['batch_id'], exp_id,
                                                "%s_total_hist.png" % (exp_id))

        logging.info("run exp %3d" % exp_num)
        main_control = MulControl()
        main_control.run_exp()  # 开始运行实验

    moniter.AppendToCsv(exp_list, all_config['exp_list_path'])
