# -*- coding: utf-8 -*-
# __AUTHOR__: GZZ
# __DATE__: 2021/04/23

"""
USAGE:

Requirements: $ pip install netCDF4

数据来源：http://www.fjhyyb.cn/Ocean863Web_MAIN/default.aspx#szyb

洋流场nc文件主要参数说明：
    以洋流文件current_hour_20200913.nc为例，本文件截取自ROMS洋流模型当天的预报数据，模型采用曲线正交网格，
    共211*301个点，每个文件以UTC00时为起点，上面文件是2020年9月13日01时（对应北京时间09时）包含24个小时的预报数据。
    主要关注时间、位置及流速。
        * ocean_time：时间 seconds since 1900-01-01 00:00:00
            注意：时间在不同平台上可能差异，请参考文件名，当日北京时09时为起点，每条记录间隔1小时
        * lon：网格中心位置经度
        * lat：网格中心位置纬度
        * u: 网格中心流速的东向向量
        * v: 网格中心流速的北向向量

风场nc文件主要参数说明：
    以风场文件wind_hour_2020040212.nc为例，本文件是截取自WRF风场模型的当天的预报数据，分辨率20km一个网格点，
    共51*51个点，每个文件以UTC12时为起点，文件是2020年4月2日12时（对应北京时间20时）包含169个小时的预报数据。
    主要关注时间、位置及风速。
        * TIME：时间 minutes since 2020-04-02 00:00:00 （在漂移中需要注意加8个小时以匹配时间）
            注意：时间可以参考文件名，当日北京时20时为起点，每条记录间隔为1个小时
        * LONGITUDE51_101： 经度
        * LATITUDE51_101 ：纬度
        * U10：风的东向分量
        * V10： 风的北向分量
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure',autolayout = True)

"""
读取nc文件
"""

# TODO: 更改为合适路径
nc_file_current = './data/raw_data/otherdatav3/current/current_hour_20191130.nc'
nc_file_wind = './data/raw_data/otherdatav3/wind/wind_hour_2019121612.nc'

# 数据读入
nc_ocean=Dataset(nc_file_current)
nc_wind=Dataset(nc_file_wind)
print(nc_wind.variables)
print('ocean_time:  ', nc_ocean.variables['ocean_time'][:])
print('wind_time:  ', nc_wind.variables['TIME'][:])
"""
可视化风场nc文件
"""
# date = "2019-11-30"
date_str = list(nc_file_wind.split('/')[-1].split('.')[0].split('_')[2][:8])
date_str.insert(4, '-')
date_str.insert(7, '-')
date = ''.join(date_str)

y = nc_wind.variables['LONGITUDE1_151']
y = np.array(y)

x = nc_wind.variables['LATITUDE1_151']
x = np.array(x)

u = nc_wind.variables['U10']
u = np.array(u)
u = u[0, :, :]
u = np.deg2rad(u)

v = nc_wind.variables['V10']
v = np.array(v)
v = v[0, :, :]
v = np.deg2rad(v)

plt.figure(figsize=(16, 9))

plt.grid()
plt.xlim((min(x), max(x)))
plt.ylim((min(y), max(y)))

plt.quiver(
    x, y, u, v,
    [i for i in range(len(x))],
    pivot='tail',
)

plt.savefig('wind_hour_'+str(date)+'.jpg', dpi=600)
plt.show()
