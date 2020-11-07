clc;close all;clear all;
a = 1; b = 1;
m0 = 2;
m1 = 20;
m2 = 1;

w1 = rand(m1,m0+1);
w2 = rand(m2,m1+1);
eta1 = 0.01;
s = 0:100000;
yp=[0 0];
ym=[0 0];
u = 0;

for k = 1:length(s)
    
        p = [1;yp(end);yp(end-1)];
        r = sin(2*pi*k/25);
        
        yp1 = yp(end-1)*yp(end)*(yp(end) + 2.5)/(1 + (yp(end))^2 + (yp(end-1))^2) + u;
        
        
        ym1 = 0.6*ym(end) + 0.2*ym(end-1) + r;
       
       
        %FORWARD PASS
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k;
        y2 = v2 ;
        E =  yp1 - u - y2;
        
        phi_v2_diff = (1);
        phi_v1_diff =  (b/a)*(a^2 - (phi_v1.^2)); 
       
   
        %BACKWARD PASS
    
        delta2 = E*phi_v2_diff;  
        delta_w2 = eta1.*delta2*(y1_k)'; 
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION

        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;         
    
        u = -y2 + 0.6*yp(end) + 0.2*yp(end-1) + r;
        yp=[yp yp1];
        ym=[ym ym1];
        
        
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% -----------------------------------------------------------------------

K = 0:length(s)+1;
figure;
plot(K,yp,'-r',K,ym,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 7 with control - Single Layer BPA');
axis([0 500 -4 6]);

g=var(ym(3:end)-yp(3:end));
h=var(ym(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);
