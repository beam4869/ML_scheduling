using  StatsBase, CSV, DataFrames, XLSX
df = DataFrame(XLSX.readtable("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.xlsx","Sheet1"))
# df = CSV.read("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/machine learning/datasets for ML/total dataset for ML with statistic numbers.csv",DataFrame)
groupings = df[!,"groupings"][1:21504]
groupingNum = zeros(length(groupings)).+9
for i in 1:length(groupings)
    if groupings[i] == "[[1, 2], [3, 4]]" || groupings[i] == "[[3, 4], [1, 2]]" || groupings[i] == "[[2], [1, 3, 4]]"|| groupings[i] == "[[4], [1, 2, 3]]"
        groupingNum[i] = 0
    elseif  groupings[i] == "[[1, 4], [2, 3]]" || groupings[i] == "[[2, 3], [1, 4]]" || groupings[i] == "[[1], [2, 3, 4]]"
        groupingNum[i] = 1
    end
end
df2 = DataFrame(groupingNums = groupingNum)
CSV.write("two classification dataset grouping numbers.csv", df2)