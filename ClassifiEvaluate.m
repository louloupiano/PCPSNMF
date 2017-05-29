function [ConfuMatrix,CorrectRate,Kappa]=ClassifiEvaluate(MatrixClassTable,Class)
%求分类混淆矩阵ConfuMatrix，总体精度CorrectRate，Kappa系数
%MatrixClassTable为列的矩阵，第一列为目标类别，第二列为计算出的类别
%类别数为Class
%混淆矩阵定义来自孙家柄《遥感原理》
%总体精度CorrectRate，Kappa系数定义来自文章《Status of land cover classification accuracy assessment 》
%Remote Sensing of Environment,2002,80:185-201
MatrixClassTable=double(MatrixClassTable);
ClassNum=Class;
ConfuMatrix=zeros(ClassNum,ClassNum);
CorrectRate=0;
Kappa=0;
%混淆矩阵每行的和
RowSum=zeros(ClassNum,1);
%混淆矩阵每列的和
ColSum=zeros(1,ClassNum);
[Row,Col]=size(MatrixClassTable);
%计算混淆矩阵
for i=1:Row
    ConfuMatrix(MatrixClassTable(i,1),MatrixClassTable(i,2))=ConfuMatrix(MatrixClassTable(i,1),MatrixClassTable(i,2))+1;
end
%计算总体精度CorrectRate
%计算Kappa系数
for i=1:ClassNum
    for j=1:ClassNum
         RowSum(i,1)=RowSum(i,1)+ConfuMatrix(i,j);
         ColSum(1,j)=ColSum(1,j)+ConfuMatrix(i,j);
    end
end
temp1=0;
temp2=0;
for i=1:ClassNum
    temp1=temp1+ConfuMatrix(i,i);
    temp2=temp2+RowSum(i,1)*ColSum(1,i);
end
CorrectRate=temp1/Row;
Kappa=(Row*temp1-temp2)/(Row*Row-temp2);