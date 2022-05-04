%% 初始化参数
clear all;
close all;
clc;
%% 读取两点的坐标
coordinate_a = [22.476835,113.8838617];
coordinate_b = [22.48331167,113.85454];
number_b = size(coordinate_b,1);
%% 计算两点间的距离
distance_ab = [];
for i = 1:number_b
    % diatance_a的第一列的单位：度
    distance_ab(i,1) = distance(coordinate_a(1),coordinate_a(2),coordinate_b(i,1),coordinate_b(i,2))
    % diatance_ab_2的第二列单位：千米
    distance_ab(i,2) = distance(coordinate_a(1),coordinate_a(2),coordinate_b(i,1),coordinate_b(i,2))/180*pi*6371
end