#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import yaml
import os


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def load_json(file_path):
    '''
    description: 加载json文件
    return {ouput}
    '''    
    with open(file_path, 'r') as f:
        output = json.load(f)
    return output


def format_print_dict(print_dict):
    print_str = ''
    len_print_dict = len(print_dict.keys())
    for i, k in enumerate(print_dict.keys()):
        print_str += f'{k}: '
        if isinstance(print_dict[k], int):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k]}, '
            else:
                print_str += f'{print_dict[k]}'
        elif isinstance(print_dict[k], float):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k] :.5f}, '
            else:
                print_str += f'{print_dict[k] :.5f}'
        elif isinstance(print_dict[k], str):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k]}, '
            else:
                print_str += f'{print_dict[k]}'
        elif isinstance(print_dict[k], list):
            print_str += '['
            v_list = print_dict[k]
            lth = len(v_list)
            for i in range(lth):
                if i == lth - 1:
                    print_str += f'{print_dict[k][i] :.5f} '
                    print_str += ']'
                else:
                    print_str += f'{print_dict[k][i] :.5f}, '

    return print_str
