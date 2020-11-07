clc;close all;clear all;

a = 1; b = 1;
m0 = 3;
m1 = 20;
m2 = 1;

w1 = rand(m1,m0+1);
w2 = zeros(m2,m1+1);
eta = 0.001;
n = 0:100000;
yp=[0 0 0];
ym=[0 0 0];
u=[0 0];


for k = 1:length(n)
    
        p = [1;yp(end);yp(end-1);yp(end-2)];
        r = sin(2*pi*k/25);  
        
        yp1 =(5*yp(end)*yp(end-1)/(1 + (yp(end))^2 + (yp(end-1))^2+ (yp(end-2))^2))+ u(end)+ 1.1*u(end-1);
        yp=[yp yp1];
         
        ym1 = 0.32*ym(end) + 0.64*ym(end-1)-0.5*ym(end-2)+ r;
        ym=[ym ym1];
              
        %FORWARD PASS

        v1 = w1*p; 
        phi_v1 = a*tanh(b*v1); 
        y1_k = [1 ; phi_v1];
        v2 = w2*y1_k;
        y2 = v2 ;
        
        E  =(5*yp(end)*yp(end-1)/(1 + (yp(end))^2 + (yp(end-1))^2+ (yp(end-2))^2)) - y2;
        phi_v2_diff = (1);
        phi_v1_diff =  (b/a)*(a^2 - (phi_v1.^2)); 
    
        
        %BACKWARD PASS

        delta2 = E*phi_v2_diff;  
        delta_w2 = eta.*delta2*(y1_k)'; 
        
        delta1 = (w2(:,2:length(w2))'*delta2).*(phi_v1_diff);
        delta_w1 = eta.*delta1*(p)';
             
        
        %ERROR CALCULATION AND WEIGHT UPDATION

        w1 = w1 + delta_w1;
        w2 = w2 + delta_w2;    
        

        u1 = (-y2 -(1.1)*u(end)+0.32*yp(end) + 0.64*yp(end-1)-0.5*yp(end-2)+ r);
      
        u = [u u1];


end
% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K = 0:length(n)+2;
figure;
plot(K,yp,'-r',K,ym,'--g')
axis([0 50  -7 7])
legend('Plant Output', 'Reference Model Output');
title('Example 9 with control - Single layer');

g=var(yp(3:end)-ym(3:end));
h=var(ym(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp(3:end)-ym(3:end)).^2)/(length(yp(3:end)));
disp(mse);
K=0:length(n)+1;
figure;
plot(K,u)
axis([0 50 -300 300])
title('Example 9 oscillatory input');