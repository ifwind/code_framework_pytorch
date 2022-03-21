# ！/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "L"

# -*- coding: utf-8 -*-

import logging


def get_logger(output_file):
    # get TF logger
    logger = logging.getLogger('logger_name')
    logger.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(output_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
