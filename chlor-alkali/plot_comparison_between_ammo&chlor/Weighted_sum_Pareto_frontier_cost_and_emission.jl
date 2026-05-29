using JuMP, CPLEX, Plots, StatsBase, CSV, DataFrames, XLSX, DelimitedFiles
t=zeros(48)
wind=zeros(48)
wpow=zeros(48) # the amount of power produced by wind.
vci = 3
vr = 7
vco = 12
pwr = 15000 #average the capacity factors!
for i in 1:length(wind)
    t[i] = i
    wind[i]=0
    wpow[i] = 0

end
avgwind=mean(wind)
#display(plot(t,[wpow], labels = ["Wind Speed, m/s" "Power Generated, MW"]))
tl=length(t)
il = 3 # the number of three main materials: H2 N2 NH3;
rho = [60 0.8 2.12] #linear conversion factor of three materials
mmin = [9.99 0 750] # min production speed of an electrolyzer of three materials. 
mmax = [33.3 2250 1100] # max production speed of an electrolyzer of three materials. 
nemax = 15
msmin = [4035 22601 0]
msmax = [807197 4.52e6 0]
mnh3tar = 1000 #the NH3 production target
rmin = 100
rmax = 40
rampt = 4
damax = 1
ceb = zeros(48)
gridemm=zeros(48)
ces = zeros(48)
cnh3 = 1.46 # the NH3 price 
con = 50
cel = 0.42
cpsa = .10 # the cost of N2 from electrolyzer
csh2 = 5.50 
csh2 = 5.50 # price of H2
csn2 = 0.14 # price of N2
cbh2=2.30 # price of H2 from steam process
h2perh=100
nh3_tot_min = 45000

dir_initalconditions1 = "G:\\My Drive\\AA gourp\\DCE\\data constructing for DCE\\test-Jan-2022"#csv data folder directory   get  the price and emission data from the xlsx
x1 = readdir(dir_initalconditions1) #read all the name of the files in the folder
df1 = DataFrame(XLSX.readtable("G:\\My Drive\\AA gourp\\DCE\\data constructing for DCE\\test-Jan-2022\\"*x1[1],"test_for_pareto_frontier"))
# df2 = DataFrame(XLSX.readtable("I:\\AA gourp\\machine learning\\data_construction\\test_2022_New_England_Hub\\"*x1[1],"emission"))
#println(df)
ceb = df1[!,"Jan 21st 11pm price"]#entering the first line of the row to let it find it
gridemm = df1[!,"Jan 21st 11pm emission"]

ces=ceb.-0.03
h2fossil=9.3  # carbon emission factor from hydrogen derived from steam reforming of methane. units: kg CO2/Kg H2
el_start_risk=5
el_op_risk=1
reac_op_risk=1
reac_ramp_risk=2.5
stor_n2_risk=1
stor_h2_risk=1.5

mi0=copy(msmin)
ne0=0
mnh30=copy(mnh3tar)
da0=[0,0,0]
times = append!([0],t)

emiss_cost_matrix = zeros(99,2)
#optimization model
nh3w=Model(CPLEX.Optimizer)

#variables
@variable(nh3w, da[-2:tl], Bin)
@variable(nh3w, de[1:tl]>=0, Int) # the number difference of the electrolyzer with time varying.
@variable(nh3w, ms[1:il,0:tl]>=0) # amount of N2 H2 in storage
@variable(nh3w, mp[1:il,0:tl]>=0) # production rate of three materials
@variable(nh3w, mnh3dev[1:tl]) # the deviation between the target and the real production
@variable(nh3w, ne[0:tl], Int) # the number of "on" state electrolyzer
@variable(nh3w, peb[1:tl]>=0) # the amount of power bought from the grid.
@variable(nh3w, pei[1:il,1:tl]>=0) #the power used by different chemical units: H2 N2 NH3
@variable(nh3w, pes[1:tl]>=0)  # the amount of power sold to the grid.
@variable(nh3w, u[1:il]>=0)# the total depleted H2 and N2 gas during the production period.
@variable(nh3w, bh2[1:tl]>=0)  # H2 from the steam process.

#steady constraints. section 2.2
@constraint(nh3w, energybal[i in 1:tl], wpow[i]+peb[i]-pes[i]-pei[1,i]-pei[2,i]-pei[3,i]==0) # energy balance
@constraint(nh3w, energyunt[i in 1:il, j in 1:tl], pei[i,j] == rho[i]*mp[i,j]) #the power used by different chemical units is propotional to the mass of the materials
@constraint(nh3w, minflow[i in 2:il, j in 1:tl], mmin[i]<=mp[i,j]) #lowerbound of the production rate of N2, H2, NH3
@constraint(nh3w, maxflow[i in 2:il, j in 1:tl], mmax[i]>=mp[i,j]) #upperbound of the production rate of N2, H2, NH3
@constraint(nh3w, minelec[j in 1:tl], ne[j]>=0) # the number of "on" state electrolyzer should be larger than 0
@constraint(nh3w, maxelec[j in 1:tl], ne[j]<=nemax) # the number of "on" state electrolyzer can't excel the total number of the machine.
@constraint(nh3w, minh2pr[j in 1:tl], mmin[1]*ne[j]<= mp[1,j])
@constraint(nh3w, maxh2pr[j in 1:tl], mp[1,j]<=mmax[1]*ne[j]) # prduction rate of H2 is bounded by the number of "on" machine time the min/max production rate of one electrolyzer 
@constraint(nh3w, minstor[i in 1:il, j in 1:tl], msmin[i]<=ms[i,j])
@constraint(nh3w, maxstor[i in 1:il, j in 1:tl], msmax[i]>=ms[i,j]) # material storage boundary.
@constraint(nh3w, deviation[j in 1:tl], mnh3dev[j] == mnh3tar-mp[3,j]) #economic penalty for underproduction
@constraint(nh3w, maxbh2[j in 1:tl], bh2[j]<=h2perh) #upperbound of the H2 from steam process.
#@constraint(nh3w, sum(mp[3,j] for j=1:tl)>=nh3_tot_min)# difference!

#dynamic constraints, section 2.3
@constraint(nh3w, h2stor[j in 1:tl], ms[1,j]==ms[1,j-1]+bh2[j]+mp[1,j]-(3/17)*mp[3,j]) #ms[i,0] the iteration process of H2
@constraint(nh3w, n2stor[j in 1:tl], ms[2,j]==ms[2,j-1]+mp[2,j]-(14/17)*mp[3,j]) # the iteration process of N2
@constraint(nh3w, totalstor[i in 1:il], u[i]==ms[i,0]-ms[i,48]) #ms[i,0] # the total depleted H2 and N2 gas during the production period.
@constraint(nh3w, elecon[j in 1:tl], de[j]>=ne[j]-ne[j-1]) #ne[0] track the number of electrolyzers starting up in a time period se for cost and safety purposes
@constraint(nh3w, rampmin[j in 1:tl], -da[j]*rmin <= mp[3,j]-mp[3,j-1]) #mp[i,0]
@constraint(nh3w, rampmax[j in 1:tl],  mp[3,j]-mp[3,j-1] <= da[j]*rmax) #the NH3 producing rate should in a boundary.
@constraint(nh3w, ramplim[j in 1:tl], sum(da[i] for i in (j+1-rampt):j)<=damax) #da[-2,-1,0] total change should less than a constant.

#dynamic constraint feed-in data constraints
@constraint(nh3w, ms_t0[i in 1:il], ms[i,0] == mi0[i]) #make the initial into the variable.
@constraint(nh3w, ne_t0, ne[0]==ne0) #make the initial into the variable.
@constraint(nh3w, mnh3_t0, mp[3,0]==mnh30) #make the initial into the variable.
@constraint(nh3w, da_t[j in -2:0], da[j]==da0[j+3]) 

#objective expressions
@expression(nh3w, costs, sum(ceb[j]*peb[j]-ces[j]*pes[j]+cnh3*mnh3dev[j]
+con*de[j]+cel*mp[1,j]+cbh2*bh2[j]+cpsa*mp[2,j] for j in 1:tl)+csh2*u[1]+csn2*u[2])
#@expression(nh3w, emms, sum(mp[1,i]*gridemm*rho[1]+mp[2,i]*gridemm*rho[2]+mp[3,i]*gridemm*rho[3] for i in 1:tl)
#+sum(h2fossil*bh2[i] for i in 1:tl)+sum(peb[i]*gridemm for i in 1:tl))#-sum(wpow[i]*gridemm for i in 1:tl))
@expression(nh3w, emms, sum(sum(rho[j]*mp[j,i]*gridemm[i] for j in 1:il) for i in 1:tl)
+sum(h2fossil*bh2[i] for i in 1:tl))
@expression(nh3w, watuse, sum(mp[1,j]*9.0 for j in 1:tl)+sum(bh2[j]*4.5 for j in 1:tl)) # water usage; 
@expression(nh3w, safetyelectro, sum(el_op_risk*ne[i] for i in 1:tl)+sum(el_start_risk*de[i] for i in 1:tl))  #safety risk: sum of two differen kind of risk.

#epsilon constraints
#@constraint(nh3w,epcone,emms<=epve) # emmision should less than a value
#@constraint(nh3w,epconr,watuse<=epvr) # water usage should also less than a value.
for i = 1:99
    @objective(nh3w, Min, i*0.01*costs+(1-i*0.01)*(emms+watuse+safetyelectro)) #objective
    JuMP.optimize!(nh3w) # depend on the package that you are using.
    emiss_cost_matrix[i,1] = value(costs)
    emiss_cost_matrix[i,2] = value(emms)+value(watuse)+value(safetyelectro)
end
display(scatter(emiss_cost_matrix[:,1],emiss_cost_matrix[:,2],xlabel = "costs", ylabel = "emission+watuse+safetyelectro",size=[800,250],margin=10Plots.mm))

