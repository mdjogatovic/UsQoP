import sqlite3, dateparser
from pyes.util import day, microsecond
from pyes.proc import queue, resource
from scipy.interpolate import interp1d

class dbutil:
  conn = None
  def __init__(self, dbpath=""):
    # Konekcija sa bazom podataka
    if not dbutil.conn:
      dbutil.conn = sqlite3.connect(dbpath)
  
  def tariff_packages(self,scenario):
    if not isinstance(scenario,str):
      raise ValueError("dbutil.tariff_packages#scenario - string is expected")
    tariff_package = {}
    for row in dbutil.conn.execute("""SELECT Packages.Name, Packages.Price FROM Packages """
                                   """INNER JOIN Scenarios """
                                   """ON Packages.ScenarioId=Scenarios.Id """
                                   """WHERE Scenarios.Name='{0}'""".format(scenario)):
      tariff_package[row[0]] = row[1]
    return tariff_package

  def tariff_packages_freq(self,scenario,demand):
    if not isinstance(scenario,str):
      raise ValueError("dbutil.tariff_packages_freq#scenario - string is expected")
    if not isinstance(demand,str):
      raise ValueError("dbutil.tariff_packages_freq#demand - string is expected")

    tp_freq = [[],[]]
    for row in dbutil.conn.execute("""SELECT Packages.Name, PackagesDemand.Frequency, Packages.Price FROM PackagesDemand """
                                   """INNER JOIN Packages ON PackagesDemand.PackageId = Packages.Id """
                                   """INNER JOIN Demands  ON PackagesDemand.DemandId = Demands.Id """
                                   """INNER JOIN Scenarios ON Packages.ScenarioId = Scenarios.Id """
                                   """WHERE Demands.Name = '{0}' AND Scenarios.Name = '{1}'""".format(demand,scenario)):
      tp_freq[0].append((row[0],row[2]))
      tp_freq[1].append(row[1])
    return tp_freq

  def network_load_distributions(self):
    nl_dists = {}

    for row in dbutil.conn.execute("""SELECT NetworkLoadFrequency.LowerTime, NetworkLoadFrequency.UpperTime, NetworkLoad.Name, NetworkLoadFrequency.Frequency/100, NetworkLoadFrequency.MeanTime, NetworkLoad.TimeInterval, NetworkLoadFrequency.Weekend FROM NetworkLoadFrequency """
                                   """INNER JOIN NetworkLoad """
                                   """ON NetworkLoadFrequency.NetworkLoadId=NetworkLoad.Id"""):
      if not row[-1]:
        if not "workday" in nl_dists:
          nl_dists["workday"] = {}
        ti = (dateparser.parse(row[0]).time(),dateparser.parse(row[1]).time())
        if not ti in nl_dists["workday"]:
          nl_dists["workday"][ti] = [[],[],[],[]]
        nl_dists["workday"][ti][0].append(row[2])
        nl_dists["workday"][ti][1].append(row[3])
        nl_dists["workday"][ti][2].append(row[4])
        nl_dists["workday"][ti][3].append(row[5])
      else:
        if not "weekend" in nl_dists:
          nl_dists["weekend"] = {}
        ti = (dateparser.parse(row[0]).time(),dateparser.parse(row[1]).time())
        if not ti in nl_dists["weekend"]:
          nl_dists["weekend"][ti] = [[],[],[],[]]
        nl_dists["weekend"][ti][0].append(row[2])
        nl_dists["weekend"][ti][1].append(row[3])
        nl_dists["weekend"][ti][2].append(row[4])
        nl_dists["weekend"][ti][3].append(row[5])
    return nl_dists

  def demands(self):
    dems = [[],[]]
    for row in dbutil.conn.execute("SELECT Name, Frequency FROM Demands"):
      dems[0].append(row[0])
      dems[1].append(row[1])
    return dems

  def price_reduction(self,package,nl,scenario):
    if not isinstance(package,str):
      raise ValueError("dbutil.price_reductions#package - string is expected")
    if not isinstance(nl,str):
      raise ValueError("dbutil.price_reductions#nl - string is expected")
    if not isinstance(scenario,str):
      raise ValueError("dbutil.price_reductions#scenario - string is expected")

    cur = dbutil.conn.execute("""SELECT NetworkLoadPriceRed.PriceReduction FROM NetworkLoadPriceRed """
                              """INNER JOIN Packages ON NetworkLoadPriceRed.PackageId = Packages.Id """
                              """INNER JOIN NetworkLoad  ON NetworkLoadPriceRed.NetLoadId = NetworkLoad.Id """
                              """INNER JOIN Scenarios ON Packages.ScenarioId = Scenarios.Id """
                              """WHERE Packages.Name='{0}' AND NetworkLoad.Name = '{1}' AND Scenarios.Name='{2}'""".format(package,nl,scenario))
    return cur.fetchone()
      