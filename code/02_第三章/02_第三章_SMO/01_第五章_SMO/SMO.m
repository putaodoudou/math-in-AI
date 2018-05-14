% SVM using Sequential Minimal Optimization (SMO)
clear all;clc; 
data=csvread('LinearlySeprerableData.csv');
data(:,1:end-1)=zscore(data(:,1:end-1));
x = data(:,1:end-1);
y = data(:,end);
N = length(y);
C = 0.5; %Concluded after Cross-Validation
tol = 10e-5;
alpha = zeros(N,1);
bias = 0;

%  SMO Algorithm
while (1)
    changed_alphas=0;
    N=size(y,1)
    for i=1:N
        Ei=sum(alpha.*y.*K(x,x(i,:),'l'))-y(i);
        if ((Ei*y(i)<-tol) && alpha(i)<C)||(Ei*y(i) > tol && (alpha(i) > 0))
            for j=[1:i-1,i+1:N]
                Ej=sum(alpha.*y.*(x*x(j,:)'))-y(j);
                  alpha_iold=alpha(i);
                  alpha_jold=alpha(j);

                  if y(i)~=y(j)
                      L=max(0,alpha(j)-alpha(i));
                      H=min(C,C+alpha(j)-alpha(i));
                  else 
                      L=max(0,alpha(i)+alpha(j)-C);
                      H=min(C,alpha(i)+alpha(j));
                  end

                  if (L==H)
                      continue
                  end
                  
                  eta = 2*x(j,:)*x(i,:)'-x(i,:)*x(i,:)'-x(j,:)*x(j,:)';
                  
                  if eta>=0
                      continue
                  end
                  
                  alpha(j)=alpha(j)-( y(j)*(Ei-Ej) )/eta;
                  if alpha(j) > H
                      alpha(j) = H;
                  end
                  if alpha(j) < L
                      alpha(j) = L;
                  end

                  if norm(alpha(j)-alpha_jold,2) < tol
                      continue
                  end
                  
                  alpha(i)=alpha(i)+y(i)*y(j)*(alpha_jold-alpha(j));
                  b1 = bias - Ei - y(i)*(alpha(i)-alpha_iold)*x(i,:)*x(i,:)'...
                      -y(j)*(alpha(j)-alpha_jold)*x(i,:)*x(j,:)';
                  b2 = bias - Ej - y(i)*(alpha(i)-alpha_iold)*x(i,:)*x(j,:)'...
                      -y(j)*(alpha(j)-alpha_jold)*x(j,:)*x(j,:)';
           
                 
                  if 0<alpha(i)<C
                      bias=b1;
                  elseif 0<alpha(j)<C
                      bias=b2;
                  else
                      bias=(b1+b2)/2;
                  end
                  changed_alphas=changed_alphas+1;
            end
        end
    end
    if changed_alphas==0
        break
    end
    x=x((find(alpha~=0)),:);
    y=y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end
% Weights
W=sum(alpha.*y.*x)
% Bias
bias =mean( y - x*W')
% Support Vectors
disp('Number of support Vectors : ')
disp(N)
Xsupport=x;Ysupport=y;
% Accuracy and F-measure
x=data(:,1:end-1);y=data(:,end);
fx=sign(W*x'+bias)';
[~, Accuracy, F_measure ] = confusionMatrix( y,fx )
% Plotting the Decision Boundry
hold on
scatter(x(y==1,1),x(y==1,2),'b')
scatter(x(y==-1,1),x(y==-1,2),'r')
scatter(Xsupport(Ysupport==1,1),Xsupport(Ysupport==1,2),'.b')
scatter(Xsupport(Ysupport==-1,1),Xsupport(Ysupport==-1,2),'.r')
syms x1 x2
fn=vpa((-bias-W(1)*x1)/W(2),6);
fn1=vpa((-1-bias-W(1)*x1)/W(2),6);
fn2=vpa((1-bias-W(1)*x1)/W(2),6);
fplot(fn,'Linewidth',2);
fplot(fn1,'Linewidth',1);
fplot(fn2,'Linewidth',1);
axis([-2 2 -2 2])
xlabel ('Positive Class: blue, Negative Class: red')
hold off
