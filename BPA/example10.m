clc;
close all;
clear all;

a = 1; b = 1;
m0 = 2;
m1 = 50;
m2 = 50;
m3 = 1;

w11 = 0.1*rand(m1,m0+1);
w21 = 0.1*rand(m2,m1+1);
w31 = 0.1*rand(m3,m2+1);

w12 = 0.1*rand(m1,m0+1);
w22 = 0.1*rand(m2,m1+1);
w32 = 0.1*rand(m3,m2+1);

eta1 = 0.011;
eta2 = 0.001;
yp1 = 0;
yp2 = 0;
ym1 = 0;
ym2 = 0;
u1 = 0;
u2 = 0;
s = 0:10000;

for k = 1:length(s)
    
        p = [1;yp1(end);yp2(end)];
        
        r1=sin(2*pi*k/25);
        r2=cos(2*pi*k/25);
        
        yp1_k = yp1(end)/(1+yp2(end)^2) + u1;
         
        ym1_k = 0.6*ym1(end) + 0.2*ym2(end) + r1;
           
        yp2_k = (yp1(end)*yp2(end))/(1 + (yp2(end))^2) + u2;
            
        ym2_k = 0.1*ym1(end) - 0.8*ym2(end) + r2;
        
        
        %FORWARD PASS for N1
        v11 = w11*p; 
        phi_v11 = a*tanh(b*v11); 
        y11_k = [1 ; phi_v11];
        v21 = w21*y11_k;
        phi_v21 = a*tanh(b*v21);
        y21_k = [1 ; phi_v21];
        v31 = w31*y21_k; 
        y31 = v31 ;
        
        E1 =  yp1(end)/(1+yp2(end)^2) - y31;
        phi_v31_diff = (1);
        phi_v21_diff =  (b/a)*(a^2 - (phi_v21.^2)); 
        phi_v11_diff =  (b/a)*(a^2 - (phi_v11.^2)); 
    
        
        %BACKWARD PASS FOR N1
    
        delta31 = E1*phi_v31_diff;  
        delta_w31 = eta1.*delta31*(y21_k)'; 
        delta21 = (w31(2:length(w31))'.*delta31).*(phi_v21_diff); 
        delta_w21 = eta1.*delta21*(y11_k)';
        delta11 = (w21(:,2:length(w21))'*delta21).*(phi_v11_diff);
        delta_w11 = eta1.*delta11*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION for N1

        w11 = w11 + delta_w11;
        w21 = w21 + delta_w21;
        w31 = w31 + delta_w31;
   
        %FORWARD PASS FOR N2
        v12 = w12*p; 
        phi_v12 = a*tanh(b*v12); 
        y12_k = [1 ; phi_v12];
        v22 = w22*y12_k;
        phi_v22 = a*tanh(b*v22);
        y22_k = [1 ; phi_v22];
        v32 = w32*y22_k; 
        y32 = v32 ;
        E2 = (yp1(end)*yp2(end))/(1 + (yp2(end))^2) - y32;
        phi_v32_diff = 1;
        phi_v22_diff =  (b/a)*(a^2 - (phi_v22.^2)); 
        phi_v12_diff =  (b/a)*(a^2 - (phi_v12.^2)); 
   
       
        %BACKWARD PASS FOR N2
        delta32 = E2*phi_v32_diff;  
        delta_w32 = eta1.*delta32*(y22_k)'; 
        delta22 = (w32(2:length(w32))'.*delta32).*(phi_v22_diff); 
        delta_w22 = eta1.*delta22*(y12_k)';
        delta12 = (w22(:,2:length(w22))'*delta22).*(phi_v12_diff);
        delta_w12 = eta2.*delta12*(p)';
        
        %ERROR CALCULATION AND WEIGHT UPDATION FOR N2

        w12 = w12 + delta_w12;
        w22 = w22 + delta_w22;
        w32 = w32 + delta_w32;
        
        u1 = -y31 + 0.6*yp1(end) + 0.2*yp2(end) + r1;
        u2 = -y32 + 0.1*yp1(end) - 0.8*yp2(end) + r2;
        
        yp1 = [yp1 yp1_k];
        ym1 = [ym1 ym1_k];
        yp2 = [yp2 yp2_k];
        ym2 = [ym2 ym2_k];
        
end
        

% ----------------------------------------------------------------------- %
%                           Plotting Graphs                               %
% ----------------------------------------------------------------------- %
K = 0:length(s);
figure;
plot(K,yp1,'-r',K,ym1,'--g')
legend('Plant Output', 'Reference Model Output');
title('Example 10(a) with control action - Two Layer BPA');
axis([0 500 -3 3]);
g=var(ym1(3:end)-yp1(3:end));
disp(g)
h=var(ym1(3:end));
disp(h)
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp1(3:end)-ym1(3:end)).^2)/(length(yp1(3:end)));
disp(mse);

figure;
plot(K,yp2,'--g',K,ym2,'r')
legend('Plant Output', 'Reference Model Output');
axis([0 500 -3 3]);
title('Example 10(b) with control action');
g=var(ym2(3:end)-yp2(3:end));
h=var(ym2(3:end));
perf=(1-(g/h))*100;
disp(perf);
mse = sum((yp2(3:end)-ym2(3:end)).^2)/(length(yp2(3:end)));
disp(mse);
