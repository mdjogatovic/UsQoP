#! /usr/bin/python3
##############################################################
## User-centric QoS-based pricing model                     ##
## Authors: Vesna Radonjic-Djogatovic, Marko S. Djogatovic  ##
## Year: November, 2019                                     ##
##############################################################

from pyes.base import simulation, \
                      clock, start_time,\
                      time_unit, print_time
from pyes.util import timer, minute, day
from pyes.util import rn1, rn2, rn3, rn4
import numpy.random, time
from datetime import datetime, date, timedelta
from dbutil import dbutil
import statistics, pickle
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# Number of experiments
NE = 50
# Number of user in the sysetem
NU = 10000
# Start simulation time
ST = datetime(2019,1,1)
# Start BS
SC = "BS1"
# Upper seed value
HIGH_RN = 65536

# RNG that generates RNGs' seeds 
rn = numpy.random.RandomState(1)

# RNGs for each execution
ESEED = [(rn.randint(1,HIGH_RN),rn.randint(1,HIGH_RN),
            rn.randint(1,HIGH_RN),rn.randint(1,HIGH_RN)) for _ in range(NE)]

# Print time
print_time(True)
# Set start time of simulation
start_time(ST)
# Set simulation time unit
time_unit(minute)

# Number of days per month in 2019
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Demands
demads = None
# NL distributions
nl_dists        = None
# Tariff packages
tariff_packages = {}
# Selected users prices
tp_prices = {}
# Selected users prices at the end of every month
tp_prices_end = {}
# NL changes for different months
nl_changes = {}
# Revenue
revenue = {}
# Users in the system
users = []

# Tariff package class
class tariff_package:
  """Tariff package class"""
  def __init__(self,p):
    if not isinstance(p,float):
      raise ValueError("usqop.tariff_package.__init__#p - float is expected")
    self.__price = [p, p]

  def reduce_price(self,percent):
    self.__price[1] -= self.__price[0]*(percent/100.0)
  
  def reset_price(self):
    self.__price[1] = self.__price[0]

  @property
  def price(self):
    return self.__price[1]

  @property
  def base_price(self):
    return self.__price[0]

# User class
class user:
  "User class"
  __nid = 0
  def __init__(self,demand):
    if not isinstance(demand,str):
      raise ValueError("usqop.user.__init__#demand - string is expected")
    user.__nid += 1
    # User ID
    self.__id = user.__nid
    # Demand
    self.__demand = demand
    # Tariff package
    self.__tp     = None

  @property
  def id(self):
    return self.__id

  @property
  def demand(self):
    return self.__demand

  @property
  def tariff_package(self):
    return self.__tp

  @tariff_package.setter
  def tariff_package(self,tp):
    if not isinstance(tp,tariff_package):
      raise ValueError("usqop.user.tariff_package#tp - tariff_package object is expected")
    # Users TP
    self.__tp = tp

  def __str__(self):
    return "{0}".format(self.__id)

# Set users with different demands to the population
def populate_users():
  global users
  users = []
  for _ in range(NU):
    # We select users' demands
    demand = rn1.choice(demands[0],p=demands[1])
    # We create the user and place him in the population
    users.append(user(demand))

# Select TPs for users
def users_tp():
  # We iterate throught the population of users
  for user in users:
    # We return the tariff packages and the probabilities of choosing the tariff packages
    tp_freq = dbutil(DB_PATH).tariff_packages_freq(SC,user.demand)
    # We select TP for a user
    tpn = rn2.choice(len(tp_freq[0]),p=tp_freq[1])
    # We set the tariff package for the user
    user.tariff_package = tariff_packages[tp_freq[0][tpn][0]]

###############################################
## EVENTS                                    ##
###############################################

## Schedule initial events
def init():
  """Initialization event"""
  # We schedule the first change in network load
  yield network_load_change()
  # We schedule the first monthly revenue calculation
  yield monthly_revenue(),days_in_month[clock().month-1]*day,1
  # We schedule a termination of simulation event
  yield finish(),365*day

## Network load change event
def network_load_change():
  """Network load change event"""
  global nl_changes
  # Current NL
  network_load    = None
  # Current time
  ct = clock()
  # Check for workday
  if ct.weekday() < 5:
    # NL time intervals
    periods = nl_dists["workday"].keys()
    # Find time interval
    per = next(filter(lambda p:ct.time()>=p[0] and ct.time()<=p[1],periods))
    # Select network load
    vals = nl_dists["workday"][per]
    nnl = rn3.choice(len(vals[0]),p=vals[1])
  # Check for weekend
  else:
    # NL time intervals
    periods = nl_dists["weekend"].keys()
    # Find time interval
    per = next(filter(lambda p:ct.time()>=p[0] and ct.time()<=p[1],periods))
    # Select network loa
    vals = nl_dists["weekend"][per]
    nnl = rn3.choice(len(vals[0]),p=vals[1])
  # Selected NL
  network_load = vals[0][nnl]
  # MTNL
  network_load_mt = vals[2][nnl]
  # Prescribed time
  network_load_ti = vals[3][nnl]

  # NL time
  nlt = rn4.exponential(network_load_mt)

  # Compare NL time to a prescribed time
  if nlt > network_load_ti:
    # Remember network load event
    if network_load in ['NL2','NL3','NL4']:
      nl_changes[SC][ct.month][0].append(network_load)
      nl_changes[SC][ct.month][1].append(nlt)
      for tp in tariff_packages:
        red = dbutil(DB_PATH).price_reduction(tp,network_load,SC)
        # If there is reduction for current network load
        if red:
          # Reduce tariff package price
          tariff_packages[tp].reduce_price(red[0])
          # Save changes
          tp_prices[SC][tp].append((clock(),tariff_packages[tp].price))

  # Schedule next network load event
  yield network_load_change(),nlt*minute

# Monthly revenue calculation event
def monthly_revenue():
  """Monthly revenue calculation event"""
  # We calculate and remember total revenue
  revenue[SC][(clock()-1*day).strftime("%b")].append(sum([user.tariff_package.price for user in users]))
  # We return user prices to the values determined by the selected TP
  for tp in tariff_packages:
    # Save reduced value
    tp_prices_end[SC][tp][clock().month].append(tariff_packages[tp].price)
    # Reset price
    tariff_packages[tp].reset_price()
    # Save changes
    tp_prices[SC][tp].append((clock(),tariff_packages[tp].price))
  # We schedule the next event at the end of the month
  yield monthly_revenue(),days_in_month[clock().month-1]*day,1

## Termination of the simulation event
def finish():
  """Termination of the simulation event"""
  # Stop simulation
  yield simulation.stop()

# Initialize model parameters
def init_parameters():
  global nl_dists, demands
  # Demands
  demands = dbutil(DB_PATH).demands()  
  # NL distributions
  nl_dists = dbutil(DB_PATH).network_load_distributions()
  # Create population of users
  populate_users()

# Initialize TPs
def init_tps():
  global tariff_packages
  # TPs
  tps = dbutil(DB_PATH).tariff_packages(SC)
  # Clear tariff packages
  tariff_packages = {}
  # Creating tariff packages
  for tp in tps:
    tariff_packages[tp] = tariff_package(tps[tp])

###############################################

## Simulation experiment
def experiment():
  global rn1, rn2, rn3, rn4, revenue, tp_prices,\
        tp_prices_end, SC, nl_changes
  # Revenue initialization
  revenue[SC] = {}
  mday = ST
  for i in range(12):
    mday += days_in_month[i]*day
    revenue[SC][mday.strftime("%b")] = []
  
  # Assigning TPs to users
  users_tp()
  # Initialize TP end of month prices statistic
  tp_prices_end[SC] = {} 
  # For each tariff package
  for tp in tariff_packages:
    tp_prices_end[SC][tp] = {}
    for m in range(1,13):
      tp_prices_end[SC][tp][m] = []
  # Initialize NL changes statistics
  nl_changes[SC] = {}
  for m in range(1,13):
    nl_changes[SC][m] = ([],[])
  # We simulate a system with multiple repetitions.
  for i in range(NE):
    # Initialize TP prices statistic
    tp_prices[SC] = {}
    # For each tariff package
    for tp in tariff_packages:
      tp_prices[SC][tp] = [(ST,tariff_packages[tp].price)]
    # RNGs with different seed
    rn1 = numpy.random.RandomState(ESEED[i][0])
    rn2 = numpy.random.RandomState(ESEED[i][1])
    rn3 = numpy.random.RandomState(ESEED[i][2])
    rn4 = numpy.random.RandomState(ESEED[i][3])
    # We run a simulation
    # Timer object
    t = timer()
    # Start measuring
    t.tic()
    # Simulation object
    sim = simulation()
    # Start simulation
    sim.start(init)
    # End measurement
    t.toc()
  pass

# BW plots
def bw_plots(tag=""):
  months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  users_tp = {"BS1": ["TP1","TP2","TP5","TP8","TP10","TP11","TP12"],"BS2":["TP1","TP2","TP5","TP8","TP9"],
              "BS3":["TP1","TP2","TP5","TP6"],"BS4":["TP1","TP2","TP4","TP5","TP6"]}
  markers  = ['o', 's', 'v', '*', 'd', 'x', '+','^','p','1','h','|']
  linestyles = ['-','--','-.',':']
  colors = ['black','gray','darkgray','lightgray']
  fig, ax = plt.subplots()
  ax.xaxis.set_major_locator(mdates.MonthLocator())
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
  for i, bs in enumerate(revenue.keys()):
    mean_revenue = []
    stdev_revenue = []
    for month in months:
      mean_revenue.append(statistics.mean(revenue[bs][month]))
      stdev_revenue.append(statistics.stdev(revenue[bs][month]))
    ax.plot([ST+sum(days_in_month[0:i])*day for i in range(1,13)],mean_revenue,\
                label=bs,color='k',linestyle=linestyles[i],marker=markers[i])
  ax.set_xlim(ST, ST+365*day)
  ax.set_xlabel("Time (month)")
  ax.set_ylabel("Revenue (m.u.)")
  #ax.set_title("ISP Revenue")
  fig.autofmt_xdate()
  plt.grid()
  #plt.tight_layout()
  plt.legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5, 1.12))
  plt.savefig("../figs/revenue"+tag+".png",dpi=300)

  for bs in users_tp.keys():
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for i, tp in enumerate(users_tp[bs]):
      insts = [p[0] for p in tp_prices[bs][tp]]
      prices = [p[1] for p in tp_prices[bs][tp]]
      ax.plot(insts,prices,label=tp,linestyle=linestyles[i%4],color=colors[i%4],marker=markers[i])
    ax.set_xlim(ST, ST+365*day)
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("Price (m.u.)")
    #ax.set_title("Service price per user - {0}".format(bs))
    fig.autofmt_xdate()
    plt.grid()
    if bs == "BS1":
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12),handletextpad=0.2,columnspacing=0.7)
    else:
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12))
    #plt.tight_layout()
    plt.savefig("../figs/price_{0}{1}.png".format(bs,tag),dpi=300)
  #plt.show()
  pass

# Color plots
def color_plots(tag=""):
  months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  users_tp = {"BS1": ["TP1","TP2","TP5","TP8","TP10","TP11","TP12"],"BS2":["TP1","TP2","TP5","TP8","TP9"],
              "BS3":["TP1","TP2","TP5","TP6"],"BS4":["TP1","TP2","TP4","TP5","TP6"]}
  markers  = ['o', 's', 'v', '*', 'd', 'x', '+','^','p','1','h','|']
  linestyles = ['-','--','-.',':']
  colors = {"BS1":[(0.09047371119989064, 0.05445716110232384, 0.13852748506080714),
                   (0.09969720832685408, 0.15801554750586727, 0.2709350859808844),
                   (0.2859128565523805, 0.4710624660630873, 0.18738893702815024),
                   (0.8256488974926798, 0.5070500942798608, 0.6257479983967315),
                   (0.769116287350359, 0.7357927626253576, 0.9434086075123853), 
                   (0.7692547706801687, 0.8676949271654195, 0.948566973832207), 
                   (0.8640896943830253, 0.9544571010212727, 0.9375776402960758)],
            "BS2":[(0.10120483036935347, 0.08028952282539623, 0.18460915180334375),
                   (0.08605633600581403, 0.23824692404212, 0.30561236308077167),
                   (0.6328422475018423, 0.4747981096220677, 0.29070209208025455),
                   (0.7587183008012618, 0.7922069335474338, 0.9543861221913403), 
                   (0.826811144552662, 0.9338331128274076, 0.9359622361392606)],
            "BS3":[(0.10231025194333628, 0.13952898866828906, 0.2560120319409181),
                   (0.10594361078604106, 0.3809739011595331, 0.27015111282899046),
                   (0.7829183382530567, 0.48158303462490826, 0.48672451968362596),
                   (0.8046168329276406, 0.6365733569301846, 0.8796578402926125), 
                   (0.7775608374378459, 0.8840392521212448, 0.9452007992345052)],
            "BS4":[(0.10231025194333628, 0.13952898866828906, 0.2560120319409181),
                   (0.10594361078604106, 0.3809739011595331, 0.27015111282899046),
                   (0.7829183382530567, 0.48158303462490826, 0.48672451968362596),
                   (0.8046168329276406, 0.6365733569301846, 0.8796578402926125), 
                   (0.7775608374378459, 0.8840392521212448, 0.9452007992345052)]}
  nl_color = {"NL2":"navy","NL3":"orchid","NL4":"wheat"}
  plt.style.use('seaborn-dark-palette')
  fig, ax = plt.subplots()
  ax.xaxis.set_major_locator(mdates.MonthLocator())
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
  for i, bs in enumerate(revenue.keys()):
    mean_revenue = []
    stdev_revenue = []
    for month in months:
      mean_revenue.append(statistics.mean(revenue[bs][month]))
      stdev_revenue.append(statistics.stdev(revenue[bs][month]))
    ax.plot([ST+sum(days_in_month[0:i])*day for i in range(1,13)],mean_revenue,\
                label=bs,linestyle=linestyles[i],marker=markers[i])
  ax.set_xlim(ST, ST+365*day)
  ax.set_xlabel("Time (month)")
  ax.set_ylabel("Revenue (m.u.)")
  #ax.set_title("ISP Revenue")
  fig.autofmt_xdate()
  plt.grid()
  #plt.tight_layout()
  plt.subplots_adjust(left=0.14,bottom=0.13,right=0.97)
  plt.legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5, 1.12))
  plt.savefig("../figs/revenue"+tag+".png",dpi=300)

  for bs in users_tp.keys():
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for i, tp in enumerate(users_tp[bs]):
      insts = [p[0] for p in tp_prices[bs][tp]]
      prices = [p[1] for p in tp_prices[bs][tp]]
      ax.plot(insts,prices,label=tp,color=colors[bs][i],marker=markers[i])
    ax.set_xlim(ST, ST+365*day)
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("Price (m.u.)")
    #ax.set_title("Service price per user - {0}".format(bs))
    fig.autofmt_xdate()
    plt.grid()
    if bs == "BS1":
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12),handletextpad=0.2,columnspacing=0.7)
    else:
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12))
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1,bottom=0.13,right=0.97)
    plt.savefig("../figs/price_{0}{1}.png".format(bs,tag),dpi=300)
  #plt.show()
  for bs in users_tp.keys():
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for i, tp in enumerate(users_tp[bs]):
      insts = [p[0] for p in tp_prices[bs][tp]]
      prices = [p[1] for p in tp_prices[bs][tp]]
      ax.plot(insts,prices,label=tp,color=colors[bs][i],marker=markers[i])
    ax.set_xlim(ST, ST+365*day)
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("Price (m.u.)")
    #ax.set_title("Service price per user - {0}".format(bs))
    fig.autofmt_xdate()
    plt.grid()
    if bs == "BS1":
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12),handletextpad=0.2,columnspacing=0.7)
    else:
      plt.legend(loc="upper center",ncol=len(users_tp[bs]),bbox_to_anchor=(0.5, 1.12))
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1,bottom=0.13,right=0.97)
    plt.savefig("../figs/price_{0}{1}.png".format(bs,tag),dpi=300)    
    
    for bs in users_tp.keys():
      fig, ax = plt.subplots()
      for i,nl in enumerate(["NL2","NL3","NL4"]):
        y = []
        for j in range(1,13):
          y.append(len(list(filter(lambda n: n==nl, nl_changes[bs][j][0])))/NE)
        ax.bar(list(map(lambda x: x+(i-1)*0.33,range(1,13))),y,width=0.33,label=nl,color=nl_color[nl],edgecolor='black')
      ax.set_xlim(0.5,12.5)
      ax.set_xlabel("Time (month)")
      ax.set_ylabel("Average numbers of network loads")
      ax.set_xticks(range(1,13))
      ax.set_xticklabels(months,rotation=45)
      ax.set_axisbelow(True)
      plt.grid()
      plt.legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5, 1.12))
      plt.subplots_adjust(left=0.1,bottom=0.13,right=0.97)
      plt.savefig("../figs/nl_{0}{1}.png".format(bs,tag),dpi=300)

    for bs in users_tp.keys():
      fig, ax = plt.subplots()
      for i,nl in enumerate(["NL2","NL3","NL4"]):
        y = []
        for j in range(1,13):
          y.append(sum([n[1] for n in zip(nl_changes[bs][j][0],nl_changes[bs][j][1]) if n[0]==nl])/NE)
        ax.bar(list(map(lambda x: x+(i-1)*0.33,range(1,13))),y,width=0.33,label=nl,color=nl_color[nl],edgecolor='black')
      ax.set_xlim(0.5,12.5)
      ax.set_xlabel("Time (month)")
      ax.set_ylabel("Mean time of network loads (min)")
      ax.set_xticks(range(1,13))
      ax.set_xticklabels(months,rotation=45)
      ax.set_axisbelow(True)
      plt.grid()
      plt.legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5, 1.12))
      plt.subplots_adjust(left=0.1,bottom=0.13,right=0.97)
      plt.savefig("../figs/nlt_{0}{1}.png".format(bs,tag),dpi=300)
  pass

def tp_price_reduction(dbpath,filename):
  global DB_PATH,tp_prices_end
  # DB path
  DB_PATH = dbpath

  scenarios = ["BS1","BS2","BS3","BS4"]
  fin = open(filename,'rb')
  _, _, tp_prices_end, _ = pickle.load(fin)

  tp_price_red = {}
  for bs in scenarios:
    tp_price_red[bs] = {}
    tps = dbutil(DB_PATH).tariff_packages(bs)
    for tp in tps:
      tp_price_red[bs][tp] = 0.0
      for m in range(1,13):
        tp_price_red[bs][tp] += sum([(tps[tp]-v)/tps[tp] for v in tp_prices_end[bs][tp][m]])/NE
      tp_price_red[bs][tp] /= 0.12
  for bs in tp_price_red:
    print('* '+bs+' '+15*'*')
    print("{:<8} {:<15}".format('TP','Reduction (%)'))
    for k, v in tp_price_red[bs].items():
      print("{:<8} {:<15.2f}".format(k, v))
  pass

## Simulate model for different bussines scenarios and save results
def simulate_model(dbpath,filename=""):
  global DB_PATH, SC, revenue, tp_prices, tp_prices_end, nl_changes
  # DB path
  DB_PATH = dbpath
  # Model parameters initialization (demand, network load)
  init_parameters()
  # *** Experiments on different business scenarios ***
  # Scenario
  SC = "BS1"
  # TPs initialization
  init_tps()
  # Experiment
  experiment()
  # Scenario
  SC = "BS2"
  # TPs initialization
  init_tps()
  # Experiment
  experiment()
  # Scenario
  SC = "BS3"
  # TPs initialization
  init_tps()
  # Experiment
  experiment()
  # Scenario
  SC = "BS4"
  # TPs initialization
  init_tps()
  # Experiment
  experiment()

  # Save results
  fout = None
  if filename=="":
    fout = open("../results/rez_{0}.rez".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")),'wb')
  else:
    fout = open(filename,'wb')
  pickle.dump((revenue,tp_prices,tp_prices_end, nl_changes),fout)
  fout.close()

# Plot diagram
def plot_results(filename,color=True,tag=""):
  global revenue, tp_prices, tp_prices_end, nl_changes
  fin = open(filename,'rb')
  revenue, tp_prices, tp_prices_end, nl_changes = pickle.load(fin)
  fin.close()
  # Plot results
  if color:
    color_plots(tag)  
  else:
    bw_plots(tag)  

if __name__=="__main__":
  # Main menu
  print("----------------------------------------------------------")
  print("------------------ QoS BASED PRICING ---------------------")
  print("----------------------------------------------------------")
  print("1) Simulate model with users sensitive to price          |")
  print("2) Simulate model with users sensitive to quality        |")
  print("3) Plot results for model with users sensitive to price  |")
  print("4) Plot results for model with users sensitive to quality|")
  print("5) TPs price reduction percentage                        |")
  print("6) Exit                                                  |")
  print("----------------------------------------------------------")
  op = int(input("Select:"))
  if op==1:
    simulate_model("../db/qobizsim_price.db","../results/price_sensitive10000.rez")
  elif op==2:
    simulate_model("../db/qobizsim_quality.db","../results/quality_sensitive10000.rez")
  elif op==3:
    plot_results("../results/price_sensitive10000.rez",tag="_price10000")
  elif op==4:
    plot_results("../results/quality_sensitive10000.rez",tag="_qual10000")
  elif op==5:
    tp_price_reduction("../db/qobizsim_price.db","../results/price_sensitive10000.rez")
    pass
  else:
    print("See you!")
# END