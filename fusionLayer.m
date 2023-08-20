classdef fusionLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties

        % Layer properties go here
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters
       % Alpha
        % Layer learnable parameters go here
    end
    
    methods
        function layer = fusionLayer(name,name1)
            % (Optional) Create a myLayer
            % This function must have the same name as the layer
%             if nargin == 2
%                 layer.Name = name;
%             end
            layer.Name = name1;
% % 
% %             % Set layer description.
            % layer.Description = "PReLU with " + numChannels + " channels";
           % [row, col, depth,num] = size(X);
            % Initialize scaling coefficient.
          % layer.Alpha = single(ones(350,350,10,100));
        end
        
    function Z= predict(layer, X)

      %state=1;

      [row, col, depth,num] = size(X);
     tempX1 =single(ones(300,100,1,1));

      
      disp(depth)
      disp(num)
      disp(row)
      disp(col)
      %X=dlarray(x);
      for ii=1:num
            for i=1:depth
                tempX1(i,ii) = X(1,1,i,ii);
                
                %Z1=tempX;
 
               % AlphaNew(:,:,i,ii)=AAA; 
              %  Z(:,:,:,i) = Fingerprint; 
            end
            %tempX1(:,ii)=tempX;
      end
      [row1, col1, depth1,num1] = size(tempX1);
      disp(depth1)
      disp(num1)
      disp(row1)
      disp(col1)
      nn=[100;100;100];
     
      Z1 = single(ClassificationMultiClassDecFusJoint(tempX1,nn,100,3));
          %disp('00000')
      Z2=Z1;
      [row2, col2, depth2,num2] = size(Z1);
      disp(depth2)
      disp(num2)
      disp(row2)
      disp(col2)
      
      for ii=1:num
            for i=1:300
%                 if Z2(i,ii)<=0
%                     Z2(i,ii)=1;
%                 end
                tempX2(1,1,i,ii) = Z2(i,ii);
                %tempX1(:,i)=tempX;
                %Z1=tempX;
 
               % AlphaNew(:,:,i,ii)=AAA; 
              %  Z(:,:,:,i) = Fingerprint; 
            end
      end
      
      Z=tempX2;
              end

    



        function [dLdX]= backward(layer, X, Z, dLdZ, memory)
        

     
            Xzb=Z.\X;
            dLdX=Xzb.*dLdZ;   



        end
    end
end