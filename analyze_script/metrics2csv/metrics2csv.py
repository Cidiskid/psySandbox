import csv
import os
import sys
import json


def init_params():
    n_agent = 16
    n, k, p = 5, 3, 7
    t, ts = 64, 8
    exp_name = "20190114-041104"
    exp_id = "exp0"
    return n_agent, n, k, p, t, ts, exp_name, exp_id


def get_args(n_agent, n, k, p, t, ts, exp_name, exp_id):
    args_str = "NA{}_N{}_K{}_P{}_T{}_Ts{}".format(n_agent, n, k, p, t, ts)
    exp_dir = os.path.join("..", "..", "result",
                           "batch_{exp_name}_{args_str}",
                           "mul_{exp_name}_{exp_id}").format(exp_name=exp_name,
                                                             args_str=args_str,
                                                             exp_id=exp_id)
    metrics_json_name = "mul_{exp_name}_{exp_id}_metrics.json".format(exp_name=exp_name, exp_id=exp_id)
    result_csv_dir = "metrics_csv".format(exp_name=exp_name, exp_id=exp_id)
    args = {
        "metrics_json": os.path.join(exp_dir, metrics_json_name),
        "result_csv_dir": os.path.join(exp_dir, result_csv_dir),
    }
    return args


def div_table_by_path(metrics_json):
    tables = {}

    def walk_json(key_names, m_json):
        if len(list(filter(lambda s: not s.isdigit(), m_json.keys()))) == 0:
            key_name = '_'.join(key_names)
            tables[key_name] = m_json
        elif isinstance(m_json, dict):
            for key in m_json:
                walk_json(key_names + [key], m_json[key])
        else:
            print("Error!", m_json)
            raise ValueError

    walk_json([], metrics_json)
    return tables


class keypath:
    def __init__(self, data=None):
        self.d = []
        if data is None:
            self.d = []
        elif isinstance(data, str):
            self.d = keypath._keypath2keys(str)
        elif isinstance(data, list):
            self.d = data
        else:
            self.d = [data]

    def __str__(self):
        return keypath._keys2keypath(self.d)

    def __getitem__(self, item):
        return self.d[item]

    def __setitem__(self, key, value):
        self.d[key] = value

    def __len__(self):
        return len(self.d)

    def __lt__(self, other):
        i = 0
        while i < len(self) and i < len(other):
            if not type(self[i]) == type(other[i]):
                if isinstance(self[i], str) or isinstance(other[i], str):
                    return isinstance(self[i], str)
                else:
                    return self[i] < other[i]
            elif self[i] != other[i]:
                return self[i] < other[i]
            i += 1
        return len(self) < len(other)

    def __eq__(self, other):
        return str(self) == str(other)

    @staticmethod
    def _keys2keypath(keys):
        ret_s = ""
        for i in range(len(keys)):
            if isinstance(keys[i], str):
                ret_s += "." + keys[i]
            else:
                ret_s += "#" + str(keys[i])
        return ret_s

    @staticmethod
    def _keypath2keys(keypath):
        keys = []
        for s in keypath.split('.'):
            ss = s.split('#')
            if ss[0] != "":
                keys.append(ss[0])
            for i in range(1, len(ss)):
                keys.append(int(ss[i]))
        return keys


def save_table(save_path, table_json):
    assert isinstance(table_json, dict)

    def _get_time_list():
        return list(map(str, sorted(list(map(int, table_json.keys())))))

    def _get_all_keys(m_json, iter_keys=None):
        ret_keys = []
        iter_keys = [] if iter_keys is None else iter_keys
        if isinstance(m_json, dict):
            for key in m_json:
                ret_keys += _get_all_keys(m_json[key], iter_keys + [key])
        elif isinstance(m_json, list):
            for i in range(len(m_json)):
                ret_keys += _get_all_keys(m_json[i], iter_keys + [i])
        else:
            return [keypath(iter_keys)]
        return ret_keys

    def _get_data_by_key_path(m_json, keypath):
        def __get_by_keys(mm_json, keys):
            if len(keys) == 0:
                return mm_json
            try:
                mmm_json = mm_json[keys[0]]
            except:
                return None
            return __get_by_keys(mmm_json, keys[1:])

        return __get_by_keys(m_json, list(keypath))

    def _merge_and_sort_all_keys(keys_list):
        all_keys = []
        for keys in keys_list:
            all_keys += keys
        all_keys = sorted(all_keys)
        for i in range(len(all_keys) - 1, 0, -1):
            if all_keys[i] == all_keys[i - 1]:
                del all_keys[i]
        return all_keys

    time_list = _get_time_list()
    keys_list = []
    for T in time_list:
        keys_list.append(_get_all_keys(table_json[T]))
    merged_keys = _merge_and_sort_all_keys(keys_list)
    title = ["T"] + list(map(str, merged_keys))
    csv_result = []
    for T in time_list:
        the_row = {'T': T}
        for k in merged_keys:
            the_row[str(k)] = _get_data_by_key_path(table_json[T], k)
        csv_result.append(the_row)
    with open(save_path, "w") as fp:
        csv_w = csv.DictWriter(fp, fieldnames=title, lineterminator='\n')
        csv_w.writeheader()
        csv_w.writerows(csv_result)


def main(args):
    result_csv_dir = args['result_csv_dir']
    if not os.path.isdir(result_csv_dir):
        os.makedirs(result_csv_dir)
    with open(args['metrics_json'], "r") as fp:
        metrics_json = json.load(fp)
    tables_json = div_table_by_path(metrics_json)
    for key in tables_json:
        csv_name = "metrics-%s.csv" % key
        csv_path = os.path.join(result_csv_dir, csv_name)
        save_table(csv_path, tables_json[key])


if __name__ == '__main__':
    n_agent, n, k, p, t, ts, exp_name, exp_id = init_params()
    main(get_args(n_agent, n, k, p, t, ts, exp_name, exp_id))
