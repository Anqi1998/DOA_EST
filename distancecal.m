%% ��ʼ������
clear all;
close all;
clc;
%% ��ȡ���������
coordinate_a = [22.476835,113.8838617];
coordinate_b = [22.48331167,113.85454];
number_b = size(coordinate_b,1);
%% ���������ľ���
distance_ab = [];
for i = 1:number_b
    % diatance_a�ĵ�һ�еĵ�λ����
    distance_ab(i,1) = distance(coordinate_a(1),coordinate_a(2),coordinate_b(i,1),coordinate_b(i,2))
    % diatance_ab_2�ĵڶ��е�λ��ǧ��
    distance_ab(i,2) = distance(coordinate_a(1),coordinate_a(2),coordinate_b(i,1),coordinate_b(i,2))/180*pi*6371
end