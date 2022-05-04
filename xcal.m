alpha = [30 113.8838617 155];
beita = [3 113.85454 160];
dis = [];
for i = 1:size(alpha,2)
    x = alpha(1,i);
    y = beita(1,i);
    if x<180&&y<x
        dis(1,i) = (30 * sin((180-x)./180 * pi))/sin((x-y)/180 * pi);
    else if y<180&&x<y
        dis(1,i) = (30 * sin((180-x)./180 * pi))/sin((y-x)/180 * pi);
        end
    end          
end
