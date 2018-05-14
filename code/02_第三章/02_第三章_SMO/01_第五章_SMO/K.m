function ker=K(Xtrain,x_i,k)

if k=='g'
    for i=1:size(Xtrain,1)
        ker(i,1)=exp(-norm(Xtrain(i,:)-x_i)); %gaussian Kernel
    end
elseif k=='l'
    for i=1:size(Xtrain,1)
        ker(i,1)=Xtrain(i,:)*x_i'; %linear Kernel
    end
elseif k=='p'
    for i=1:size(Xtrain,1)
        ker(i,1)=(Xtrain(i,:)*x_i').^3; %poly3 Kernel
    end
end

end