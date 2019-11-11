#! /usr/bin/python3
# User-centric QoS-based pricing model
# Authors: Vesna Radonjic-Djogatovic, Marko S. Djogatovic
# Year: 2019

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

# Broj eksperimenata
NE = 50
# Broj korisnika u sistemu
NU = 1000
# Pocetno vreme simulacije
ST = datetime(2019,1,1)
# Izabrani scenario
SC = "BS1"
# Gornja granica semena
HIGH_RN = 65536

# Generator koji generise semena generatora eksperimenata
rn = numpy.random.RandomState(1)

# Generatori slucajnih brojeva za svako od izvrsenja
ESEED = [(rn.randint(1,HIGH_RN),rn.randint(1,HIGH_RN),
            rn.randint(1,HIGH_RN),rn.randint(1,HIGH_RN)) for _ in range(NE)]

# Stampamo vremenski trenutak simulacije
print_time(True)
# Postavljamo pocetno vreme simulacije
start_time(ST)
# Vremenska jedinica simulacije
time_unit(minute)

# Broj dana u mesecu
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Zahtevi
demads = None
# Trenutno mrezno opterecenje
network_load    = None
# Raspodele mreznog opterecenja
nl_dists        = None
# Tarifni paketi u modelu
tariff_packages = None
# Selected users prices
su_prices = {}
# Revenue
revenue = {}
# Korisnici u sistemu
users = []

class user:
  def __init__(self,demand):
    if not isinstance(demand,str):
      raise ValueError("dbutil.user.__init__#demand - string is expected")
    # Korisnicki zahtevi
    self.__demand = demand

  def reduce_price(self,percent):
    self.__price[1] -= self.__price[0]*(percent/100.0)
  
  def reset_price(self):
    self.__price[1] = self.__price[0]

  @property
  def demand(self):
    return self.__demand

  @property
  def tariff_package(self):
    return self.__tp

  @tariff_package.setter
  def tariff_package(self,tp):
    if not isinstance(tp,str):
      raise ValueError("dbutil.user.tariff_package#tp - string is expected")
    # Tarifni paket korisnika
    self.__tp = tp

  @property
  def price(self):
    return self.__price[1]
  
  @price.setter
  def price(self,p):
    if not isinstance(p,float):
      raise ValueError("dbutil.user.price#p - float is expected")
    # Cena i umanjena cena tarifnog paketa
    self.__price = [p,p]

  def __str__(self):
    return "{0}".format(id(self))

def populate_users():
  global users
  users = []
  for _ in range(NU):
    # Izabiramo korisnicke zahteve
    demand = rn1.choice(demands[0],p=demands[1])
    # Kreiramo korisnika i smestamo ga u populaciju
    users.append(user(demand))
  pass

def users_tp():
  for user in users:
    # Vracamo tarifne pakete i verovatnoce izbora tarifnih paketa
    tp_freq = dbutil(DB_PATH).tariff_packages_freq(SC,user.demand)
    # Izabiramo tarifni paket
    tpn = rn2.choice(len(tp_freq[0]),p=tp_freq[1])
    # Postavljamo tarifni paket i cenu korisniku
    user.tariff_package = tp_freq[0][tpn][0]
    user.price = tp_freq[0][tpn][1]
  pass

## Rasporedjujemo pocetne dogadjaje
def init():
  """Inicijalizacija simulacije"""
  # Rasporedjujemo prvu promenu mreznog opterecenja
  yield network_load_change()
  # Rasporedjujemo dogadjaj kraja meseca
  yield monthly_revenue(),days_in_month[clock().month-1]*day,1
  # Rasporedjujemo dogadjaj kraja simulacije
  yield finish(),365*day

## Promena mreznog opterecenja
def network_load_change():
  """Dogadjaj promene mreznog opterecenja"""
  global network_load
  # Tekuce vreme
  ct = clock()
  # Ukoliko je vreme u toku radne nedelje
  if ct.weekday() < 5:
    # Vremenski periodi
    periods = nl_dists["workday"].keys()
    # Pronalazimo tekuci vremenski period
    per = next(filter(lambda p:ct.time()>=p[0] and ct.time()<=p[1],periods))
    # Odredjujemo mrezno opterecenje
    vals = nl_dists["workday"][per]
    nnl = rn3.choice(len(vals[0]),p=vals[1])
  # Ukoliko je vreme u toku vikenda
  else:
    # Vremenski periodi
    periods = nl_dists["weekend"].keys()
    # Pronalazimo tekuci vremenski period
    per = next(filter(lambda p:ct.time()>=p[0] and ct.time()<=p[1],periods))
    # Odredjujemo mrezno opterecenje
    vals = nl_dists["weekend"][per]
    nnl = rn3.choice(len(vals[0]),p=vals[1])
  # Izabrano mrezno opterecenje
  network_load = vals[0][nnl]
  # Srednje vreme izabranog mreznog opterecenja
  network_load_mt = vals[2][nnl]
  # Minimalni vremenski interval za umanjenje
  network_load_ti = vals[3][nnl]

  # Vreme trajanja izabranog mreznog opterecenja
  nlt = rn4.exponential(network_load_mt)

  if nlt >= network_load_ti:
    for user in users:
      red = dbutil(DB_PATH).price_reduction(user.tariff_package,network_load,SC)
      if red:
        user.reduce_price(red[0])
        if user == su_prices[SC][user.tariff_package][0]:
          su_prices[SC][user.tariff_package][1].append((clock(),user.price))

  # Rasporedjujemo narednu promenu mreznog opterecenja
  yield network_load_change(),nlt*minute
  pass

def monthly_revenue():
  # Izracunavamo i pamtimo ukupni prihod
  revenue[SC][(clock()-1*day).strftime("%b")].append(sum([user.price for user in users]))
  # Vracamo cene korisnika na vrednosti odredjene izabranim tarifnim paketom
  #map(lambda u:u.reset_price(),users)
  for user in users:
    user.reset_price()
    if user == su_prices[SC][user.tariff_package][0]:
      su_prices[SC][user.tariff_package][1].append((clock(),user.price))
  # Rasporedjujemo naredni dogadjaj kraja meseca
  yield monthly_revenue(),days_in_month[clock().month-1]*day,1

## Kraj simulacije
def finish():
  """Dogadjaj kraja simulacije"""
  yield simulation.stop()

def init_parameters():
  global tariff_packages, nl_dists, demands
  # Zahtevi
  demands = dbutil(DB_PATH).demands()  
  # Tarifni paketi
  tariff_packages = dbutil(DB_PATH).tariff_packages(SC)
  # Raspodele mreznog opterecenja
  nl_dists = dbutil(DB_PATH).network_load_distributions()
  # Smestamo korisnike u populaciju
  populate_users()

def experiment():
  global rn1, rn2, rn3, rn4, revenue, su_prices, SC
  # Inicijalizacija prihoda
  revenue[SC] = {}
  mday = ST
  for i in range(12):
    mday += days_in_month[i]*day
    revenue[SC][mday.strftime("%b")] = []
  
  # Dodeljujemo tarifne pakete korisnicima
  users_tp()
  # Simuliramo sistem sa vise ponavljanja
  for i in range(NE):
    # Inicijalizacija izabranih korisnika
    su_prices[SC] = {}
    # Izabiranje korisnika
    for tp in tariff_packages:
      su_prices[SC][tp] = []
      su_prices[SC][tp].append(next((user for user in users if user.tariff_package==tp),None))
      su_prices[SC][tp].append([])
      if su_prices[SC][tp][0]:
        su_prices[SC][tp][1].append((ST,su_prices[SC][tp][0].price))

    rn1 = numpy.random.RandomState(ESEED[i][0])
    rn2 = numpy.random.RandomState(ESEED[i][1])
    rn3 = numpy.random.RandomState(ESEED[i][2])
    rn4 = numpy.random.RandomState(ESEED[i][3])
    # Izvrsavamo simulaciju
    # Objekat tajmera
    t = timer()
    # Zapocinjemo merenje
    t.tic()
    # Objekat simulacije
    sim = simulation()
    # Zapocinjemo simulaciju
    sim.start(init)
    # Zavrsavamo merenje
    t.toc()
  pass

# Crno beli rezultati
def results_bw(tag=""):
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
      if su_prices[bs][tp][0]:
        insts = [p[0] for p in su_prices[bs][tp][1]]
        prices = [p[1] for p in su_prices[bs][tp][1]]
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

# Rezultati u boji
def results_color(tag=""):
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
  plt.legend(loc="upper center",ncol=4,bbox_to_anchor=(0.5, 1.12))
  plt.savefig("../figs/revenue"+tag+".png",dpi=300)

  for bs in users_tp.keys():
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for i, tp in enumerate(users_tp[bs]):
      if su_prices[bs][tp][0]:
        insts = [p[0] for p in su_prices[bs][tp][1]]
        prices = [p[1] for p in su_prices[bs][tp][1]]
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
    plt.savefig("../figs/price_{0}{1}.png".format(bs,tag),dpi=300)
  #plt.show()
  pass

def simulate_model(dbpath,filename=""):
  global DB_PATH, SC, revenue, su_prices
  # Putanja do baze
  DB_PATH = dbpath
  # Inicijalizuj parameter modela
  init_parameters()
  # Eksperimenti za razlicite scenarije
  SC = "BS1"
  experiment()
  SC = "BS2"
  experiment()
  SC = "BS3"
  experiment()
  SC = "BS4"
  experiment()

  fout = None
  if filename=="":
    fout = open("../results/rez_{0}.rez".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")),'wb')
  else:
    fout = open(filename,'wb')
  pickle.dump((revenue,su_prices),fout)
  fout.close()
  # Prikazi rezultate
  results_color()

def plot_results(filename,color=True,tag=""):
  global revenue, su_prices
  fin = open(filename,'rb')
  revenue, su_prices = pickle.load(fin)
  fin.close()
  # Prikazi rezultate
  if color:
    results_color(tag)  
  else:
    results_bw(tag)  

if __name__=="__main__":
  print("----------------------------------------------------------")
  print("------------------ QoS BASED PRICING ---------------------")
  print("----------------------------------------------------------")
  print("1) Simulate model with users sensitive to price          |")
  print("2) Simulate model with users sensitive to quality        |")
  print("3) Plot results for model with users sensitive to price  |")
  print("4) Plot results for model with users sensitive to quality|")
  print("----------------------------------------------------------")
  op = int(input("Select:"))
  if op==1:
    simulate_model("../db/qobizsim_price.db","../results/price_sensitive.rez")
  elif op==2:
    simulate_model("../db/qobizsim_quality.db","../results/quality_sensitive.rez")
  elif op==3:
    plot_results("../results/price_sensitive.rez",tag="_price")
  elif op==4:
    plot_results("../results/quality_sensitive.rez",tag="_qual")

