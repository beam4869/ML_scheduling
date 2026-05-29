using  StatsBase, CSV, DataFrames, XLSX
dir_initalconditions1 = "datasets for ML/dataset of 2022 Tokyo DA"
x1 = readdir(dir_initalconditions1) #read all the name of the files in the folder
df1 = DataFrame(XLSX.readtable("datasets for ML/dataset of 2022 Tokyo DA/price and emission intensity.xlsx","Sheet1"))
df2 = CSV.read("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/dataset of 2022 Tokyo DA",DataFrame)
TokyoMatrixforML = zeros(363*24,97)
Matrix_for_SVM[:,97] = df2[!,"Column1"]
A = df1[!,"price(dollars/kwh)"]#entering the first line of the row to let it find it
# ceb_total = convert(Array{Float64,1}, A)
B = df2[!,"Emission per require"]
# gridemm_total = convert(Array{Float64,1}, B)
for m in 1:363
    for n in 1:24
# ceb=ceb_total[n+24*m-24:n+m*24+23]
# gridemm=gridemm_total[n+24*m-24:n+m*24+23]
Matrix_for_SVM[n+24*m-24,1:48] = A[n+24*m-24:n+m*24+23]
Matrix_for_SVM[n+24*m-24,49:96] = B[n+24*m-24:n+m*24+23]

    end
end
# println(Matrix_for_SVM)
df3 = DataFrame(Matrix_for_SVM, :auto)
CSV.write("Dataset_of_2022_Tokyo_DA_for_ML.csv", df3)