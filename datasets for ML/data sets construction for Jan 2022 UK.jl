using  StatsBase, CSV, DataFrames, XLSX
# dir_initalconditions1 = "/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/dataset of 2022 Tokyo DA"
# x1 = readdir(dir_initalconditions1) #read all the name of the files in the folder
df1 = DataFrame(XLSX.readtable("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/dataset of Jan 2022 UK DA/price and emission.xlsx","Sheet1"))
df2 = CSV.read("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/dataset of Jan 2022 UK DA/total_factor_grouping_2022_UK_DA.csv",DataFrame)
UKJanMatrixforML = zeros(28*24,96)
# TokyoMatrixforML[:,97] = df2[!,"Column1"]
A = df1[!,"price(dollar/kwh)"]#entering the first line of the row to let it find it
# ceb_total = convert(Array{Float64,1}, A)
B = df1[!,"Actual Carbon Intensity (kgCO2/kWh)"]
B1 = zeros(720)
for i in 1:720
    B1[i] = Float64(B[1+(i-1)*2])
end
# gridemm_total = convert(Array{Float64,1}, B)
for m in 1:28
    for n in 1:24
# ceb=ceb_total[n+24*m-24:n+m*24+23]
# gridemm=gridemm_total[n+24*m-24:n+m*24+23]
    UKJanMatrixforML[n+24*m-24,1:48] = convert(Array{Float64,1}, A[n+24*m-24:n+m*24+23])
    UKJanMatrixforML[n+24*m-24,49:96] = B1[n+24*m-24:n+m*24+23]
    end
end     
# println(Matrix_for_SVM)
df3 = DataFrame(UKJanMatrixforML, :auto)
df = hcat(df3,df2[:,1:1])
CSV.write("Dataset_of_2022_Jan_UK_DA_for_ML.csv", df)