clc;close all;clear all;
a = 1; b = 1;
m0 = 3;
m1 = 40;
m2 = 10;
m3 = 1;

w1 = rand(m1,m0+1);
w2 = rand(m2,m1+1);
w3 = rand(m3,m2+1);
eta1 = 0.05;

yp = [0 0 0];
ym= [0 0 0];
u =[0 0];
n= 0:10000;

for k =1:length(n)
    
        p = [1;yp(end);yp(end-1);yp(end-2)];
        r1 = sin(2*pi*k/25);
        
        yp1 =(5*yp(end)*yp(end-1)/(1 + (yp(end))^2 + (yp(end-1))^2+ (yp(end-2))^2))+ u(end)+(1.1)*u(end-1);
        yp=[yp yp1];
        
        ym1 = 0.32*ym(end) + 0.64*ym(end-1)-0.5*ym(end-2)+ r1;
        ym=[ym ym1];

        %FORWARD PASS
        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k;
        phi_v2 = a*tanh(b*v2);
        y2_k = [1 ; phi_v2];
        v3 = w3*y2_k; 
        y3 = v3 ;
        E  = yp1 - y3;
        phi_v3_diff = (1);
        phi_v2_diff = (b/a)*(a^2 - phi_v2.^2); 
        phi_v1_diff = (b/a)*(a^2 - phi_v1.^2); 

        %BACKWARD PASS

        delta3 = E*phi_v3_diff;  
        delta_w3 = eta1.*delta3*(y2_k)'; 
        delta2 = (w3(2:length(w3))'.*delta3).*(phi_v2_diff); 
        delta_w2 = eta1.*delta2*(y1_k)';
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta1.*delta1*(p)';

        %ERROR CALCULATION AND WEIGHT UPDATION

        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;
        w3 = w3 + delta_w3;

        u1 = -y3 -(1.1)*u(end)+0.32*yp(end) + 0.64*yp(end-1)-0.5*yp(end-2)+ r1;
        
        u=[u u1];
end

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K =0:length(n)+2;
figure;
plot(K,yp,'-r',K,ym,'--g')
axis([0 50  -7 7])
legend('Plant Output', 'Reference Model Output');
title('Example 9 with Control - Two Layer BPA');
g=var(yp(3:end)-ym(3:end));
h=var(yp(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);

K=0:length(n)+1;
figure;
plot(K,u)
axis([0 50 -300 300])
title('Example 9 oscillatory input');