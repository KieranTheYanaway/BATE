import numpy as np
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs
import obspy
import pandas as pd
import os

def get_data(name):
    """
    Reads data from folder containing seismic waveforms
    __________________________________________________________
    name : string : location of folder containing seismic data
    __________________________________________________________
    
    returns: st, inv
    ______________________________________________________________________________________________
    st  : arraylike : array containing seismic traces which will be used for the inversion process
    inv : arraylike : array containing seismic station inventories from which latitude, longitude, and elevation will be extracted
    ______________________________________________________________________________________________________________________________
    """
    st = obspy.read(name + "/waveforms/*.mseed")
    inv = obspy.read_inventory(name + "/stations/*.xml")
    st.merge(fill_value='interpolate')
    st.detrend('demean')
    st.detrend('linear')
    st.resample(10.0)
    st.taper(0.05)
    for trace in st:
        trace.filter('bandpass', freqmin=2.4,freqmax=4.4, corners = 2, zerophase = True)
        trace.normalize()
    
    return st,inv

def get_values(st,inv):
    """
    Extracts necessary values for inversion process
    ______________________________________________________________________________________________
    st  : arraylike : array containing seismic traces which will be used for the inversion process
    inv : arraylike : array containing seismic station inventories from which latitude, longitude, and elevation will be extracted
    ______________________________________________________________________________________________________________________________
    
    returns: final_array, average_datetime
    ___________________________________________________________________________________________________________________________
    final_array      : 2D arraylike : array containing columns of latitudes, longitudes, arrival times, elevations in that order
    average_datetime : UTCDateTime  : average starting time of the traces to be used in determining actual time of trajectory surface intersection
    ______________________________________________________________________________________________________________________________________________
    """
    time_delay = np.array([])  # Initialize an empty array for time delays
    times_not = np.array([])
    final_array = np.zeros((len(st),4))
    ind = 0
    start_time_array = np.zeros(0)
    for trace in st:
        times = trace.times()  # Get the time values for the trace
        datetime_times = [trace.stats.starttime + t for t in times]
        data = trace.data  # Get the data from the trace
        data = np.abs(data)  # Take the absolute value of the data
        index = np.argmax(data)  # Find the index of the maximum absolute value
        
        # Append the corresponding time to the time_delay array
        time_delay = np.append(time_delay, datetime_times[index])
        times_not = np.append(times_not, times[index])
        start_time_array = np.append(start_time_array, datetime_times[index])
        final_array[ind,3] = times[index]
        ind += 1


    datetimes_as_int = start_time_array.astype('float')

    # Calculate the average of the datetime values (in nanoseconds)
    average_datetime_int = np.mean(datetimes_as_int)


    average_datetime = obspy.UTCDateTime(average_datetime_int)
    ind = -1
    for network in inv:
        for station in network:
            for channel in station:
                ind += 1

                # Get latitude and longitude
                latitude = station.latitude
                final_array[ind,0] = latitude
                longitude = station.longitude
                final_array[ind,1] = longitude
                final_array[ind,2] = station.elevation
    
    return final_array, average_datetime


def calc_e_Rn(lat):
    """
    Calculate the plumb line at a certain latitude, helper function for ECEF_g_to_ECEF_r() and
    ECEF_r_to_ECEF_g()
    __________________________________________________________________________
    lat : float : radians latitude : latitude at which to calculate plumb line
    __________________________________________________________________________
    
    returns : e, Rn
    __________________________________________________
    e  : float : 1      : Earth ellipsoid eccentricity
    Rn : float : meters : plumb line of Earth at latitude lat
    _________________________________________________________
    """
    a = 6378137.0 # WGS-84 Earth semimajor axis, meters
    b = 6356752.3142 # WGS-84 Earth semiminor axis, meters

    f = (a - b)/a # ellipsoid flatness
    e = np.sqrt(f*(2-f)) # eccentricity
    Rn = a / np.sqrt(1 - e * e * (np.sin(lat) ** 2)) # plumb line

    return e, Rn

def ECEF_g_to_ECEF_r(lat, long, height):
    """
    Convert a point in Earth-centered Earth-fixed geodetic coordinates (lat,long,height) to a point
    in Earth-centered Earth-fixed rectangular coordinates (x,y,z)
    __________________________________________________________________________________________
    lat    : float : radians latitude  : latitude at which to calculate rectangular coordinates
    long   : float : radians longitude : longitude at which to calculate rectangular coordinates
    height : float : meters            : elevation of point at which to calculate rectangular coordinates
    _____________________________________________________________________________________________________

    returns : (x,y,z)
    ________________________________________________________________________________________________
    x : float : meters : x coordinate of point in ECEF-r coordinates
    y : float : meters : y coordinate of point in ECEF-r coordinates
    z : float : meters : z coordinate of point in ECEF-r coordinates
    ________________________________________________________________________________________________
    """

    e,N = calc_e_Rn(lat)

    # Calc x,y,z coordinates
    x = (height + N) * np.cos(lat) * np.cos(long)
    y = (height + N) * np.cos(lat) * np.sin(long)
    z = (height + (1 - e*e) * N) * np.sin(lat)

    return (x,y,z)

def ECEF_r_to_LTP(point_xyz, olat, olong):
    """
    Convert a point in Earth-centered Earth-fixed rectangular coordinates to a point relative to a local tangent plane
    _________________________________________________________________________________
    point_xyz : (x, y, z) : (m, m, m)         : x, y, z coordinate of point in ECEF-r
    olat      : float     : radians latitude  : origin latitude of local tangent plane
    olong     : float     : radians longitude : origin longitude of local tangent plane
    ___________________________________________________________________________________

    returns : x_correct
    ___________________________________________________________________________________________________
    x_correct : [x, y, z] : [m, m, m] : x, y, z coordinates of point relative to local tangent plane
    ___________________________________________________________________________________________________
    """

    x,y,z = point_xyz # Extract coordinates from tuple
    x0,y0,z0 = ECEF_g_to_ECEF_r(olat, olong, 0) # Calculate coordinates of origin for tangent plane

    # relative point location
    xp = np.array([[x-x0],
                   [y-y0],
                   [z-z0]])
    # tangent plane 3D rotation matrix
    Rt = np.array([[-np.sin(olong),                 np.cos(olong),                0           ],
                   [-np.cos(olong) * np.sin(olat), -np.sin(olat) * np.sin(olong), np.cos(olat)],
                   [np.cos(olat) * np.cos(olong),   np.cos(olat) * np.sin(olong), np.sin(olat)]])

    x_t = Rt @ xp

    # Rotate tangent coordinates 90 degrees, +x is southing and +y is easting
    ang90 = np.pi / 2
    rot90 = np.array([[np.cos(ang90), -np.sin(ang90), 0],
                      [np.sin(ang90), np.cos(ang90),  0],
                      [0,             0,              1]])
    
    x_correct = rot90 @ x_t

    return x_correct.flatten()

def LTP_to_ECEF_r(LTP_xyz, olat, olong):
    """
    Convert a point relative to a local tangent plane to a point in Earth-centered Earth-fixed rectangular coordinates
    _______________________________________________________________________________________________________
    LTP_xyz   : (x, y, z) : (m, m, m)         : x, y, z coordinate of point relative to local tangent plane
    olat      : float     : radians latitude  : origin latitude of local tangent plane
    olong     : float     : radians longitude : origin longitude of local tangent plane
    ___________________________________________________________________________________

    returns : xe
    ______________________________________________________________________
    xe : [x, y, z] : [m, m, m] : x, y, z coordinates of point in ECEF-r
    ______________________________________________________________________
    """
    
    x,y,z = LTP_xyz # Extract coordinates of local tangent plane point
    x0,y0,z0 = ECEF_g_to_ECEF_r(olat, olong, 0) # Calculate coordinates of origin for tangent plane

    x_correct = np.array([[x],
                          [y],
                          [z]])

    x_0 = np.array([[x0],
                    [y0],
                    [z0]])

    # Rotate point -90 degrees, to +x easting and +y northing
    angn90 = -np.pi / 2
    rotn90 = np.array([[np.cos(angn90), -np.sin(angn90), 0],
                       [np.sin(angn90), np.cos(angn90),  0],
                       [0,              0,               1]])

    x_t = rotn90 @ x_correct

    # Inverse rotation from the local tangent plane back to ECEF_r coordinates
    Rt = np.array([[-np.sin(olong),                 np.cos(olong),                0           ],
                   [-np.cos(olong) * np.sin(olat), -np.sin(olat) * np.sin(olong), np.cos(olat)],
                   [np.cos(olat) * np.cos(olong),   np.cos(olat) * np.sin(olong), np.sin(olat)]])
    xe = x_0 + Rt.T @ x_t

    return xe.flatten()

def ECEF_r_to_ECEF_g(xyz):
    """
    Convert a point in Earth-centered Earth-fixed rectangular coordinates (x,y,z) to a point
    in Earth-centered Earth-fixed geodetic coordinates (lat,long,height)
    ____________________________________________________________________
    xyz : (x, y, z) : (m, m, m) : x, y, z coordinates of point in ECEF-r
    ____________________________________________________________________

    returns : lat_prev, long, h
    __________________________________________________________
    lat_prev : float : latitude of point in ECEF-g coordinates
    long     : float : longitude of point in ECEF-g coordinates
    h        : float : elevation of point in ECEF-g coordinates
    ___________________________________________________________
    """
    x,y,z = xyz

    long = np.arctan2(y,x)

    r = np.sqrt(x*x + y*y + z*z)
    p = np.sqrt(x*x + y*y)

    lat_prev = np.arctan2(p,z)

    loop_iterations = 6 # higher for more accurate convergence of latitude calculation (supposedly, 4 is good enough for centimeter accuracy though)
    
    for i in range(loop_iterations):
        e,Rn = calc_e_Rn(lat_prev)
        h = p / np.cos(lat_prev) - Rn
        lat_prev = np.arctan2(z, p * (1 - e*e*(Rn/(Rn+h))))
    
    h = p/np.cos(lat_prev) - calc_e_Rn(lat_prev)[1]

    return (lat_prev, long, h)


class Trajectory:
    """
    Trajectory Class
    ----------------
    A Trajectory object wraps the parameters returned by a TrajectoryEstimator object after running a trajectory inversion.
    It will also contain some functionality for visualization of the parameters, as well as statistics from the inversion.
    """

    def __init__(self, x0, y0, t0, v, theta, gamma, olat, olong, ssr=np.infty, time=0):
        """
        Initialize a Trajectory object with the inversion parameters.
        ________________________________________________________________________________________________
        x0    : float : kilometers southing  : x position of the trajectory intersection with the x-y plane
        y0    : float : kilometers easting   : y position of the trajectory intersection with the x-y plane
        t0    : float : seconds              : time of the trajectory intersection with the x-y plane
        v     : float : kilometers/second    : velocity of the meteoroid
        theta : float : radians              : elevation angle of the trajectory
        gamma : float : radians              : azimuthal angle of the trajectory (measured anticlockwise from South)
        olat  : float : degrees latitude     : latitude of the trajectory intersection with local-tangent-plane
        olong : float : degrees longitude    : longitude of the trajectory intersection with local-tangent-plane
        ssr   : float : seconds^2            : sum of squared residuals for inversion on this trajectory
        _________________________________________________________________________________________________________
        """
        self.x0 = x0
        self.y0 = y0
        self.t0 = t0
        self.v = v
        self.theta = theta
        self.gamma = gamma
        self.olat = olat
        self.olong = olong
        lat, long, h = ECEF_r_to_ECEF_g(LTP_to_ECEF_r((1000*x0,1000*y0,0),np.deg2rad(olat),np.deg2rad(olong)))
        self.lat = np.rad2deg(lat)
        self.long = np.rad2deg(long)
        self.h = h # Elevation of intersection, technically not real
        self.ssr = ssr
        self.time = time + t0
    
    def print_parameters(self):
        print("x0:", round(self.x0, 2), "kilometers")
        print("y0:", round(self.y0, 2), "kilometers")
        print("t0:", round(self.t0, 2), "seconds")
        print("velocity:", round(self.v, 2), "km/s")
        print("theta:", round(np.rad2deg(self.theta), 2), "degrees")
        print("gamma:", round(np.rad2deg(self.gamma), 2), "degrees")
        if self.ssr is not np.infty:
            print("sum of squared residuals:", round(self.ssr,2))

    def plot(self,stations=None):
        """
        Plots an overview of the Trajectory object
        __
        stations : (lats,longs
        """
        trajlongs = np.linspace(self.long-10,self.long,2)
        slope = np.tan(self.gamma - np.pi/2)
        intercept = self.lat - slope * self.long
        ax = plt.axes(projection=ccrs.PlateCarree())
        #ax = plt.axes(projection=ccrs.Mollweide())
        ax.coastlines()
        #ax.stock_img()
        
        minx = self.olong-5
        maxx = self.olong+5
        miny = self.olat-5
        maxy = self.olat+5
        
        ax.set_xlim((minx,maxx))
        ax.set_ylim((miny,maxy))
    
        if stations is not None:
            plt.scatter(stations[1],stations[0],marker="v",label="station",c="k")
            
        plt.scatter(self.long,self.lat, c="r",label="surface intersection")
        plt.plot(trajlongs, slope * trajlongs + intercept, linestyle="--", c="b", label="trajectory")
        plt.xticks(np.linspace(minx,maxx,5))
        plt.yticks(np.linspace(miny,maxy,5))
        plt.xlabel(r"longitude ($\degree$E)")
        plt.ylabel(r"latitude ($\degree$N)")
        plt.legend()
        

class TrajectoryGridsearchParameters:
    """
    TrajectoryGridsearchParameters class
    ------------------------------------
    A TrajectoryGridsearchParameters class easily formats the inversion gridsearch parameters for
    interface with a TrajectoryEstimator object.
    """

    def __init__(self,x0p=(-300,300,10),y0p=(-300,300,10),t0p=(0,100,10),vep=(11.91015,81.6696,3),thp=(0,90,10),gap=(0,360,10)):
        """
        Initialize a TrajectoryGridsearchParameters object with the desired parameter ranges for a gridsearch
        __________________________________________________________________________________________
        x0p : (start, stop, n) : (kilometers, kilometers, integer) : range of x0s to search across
        y0p : (start, stop, n) : (kilometers, kilometers, integer) : range of y0s to search across
        t0p : (start, stop, n) : (kilometers, kilometers, integer) : range of t0s to search across
        vep : (start, stop, n) : (km/s, km/s, integer)             : range of velocities to search across
        thp : (start, stop, n) : (degrees, degrees, integer)       : range of thetas to search across
        gap : (start, stop, n) : (degrees, degrees, integer)       : range of gammas to search across
        _____________________________________________________________________________________________
        """
        self.x0s = None
        self.y0s = None
        self.t0s = None
        self.velocities = None
        self.thetas = None
        self.gammas = None
        
        self.set_x0p(x0p)
        self.set_y0p(y0p)
        self.set_t0p(t0p)
        self.set_vep(vep)
        self.set_thp(thp)
        self.set_gap(gap)
        
    def set_x0p(self, x0p):
        x0min = x0p[0]
        x0max = x0p[1]
        x0disc = x0p[2]
        self.x0s = np.linspace(x0min,x0max,x0disc)
        
    def set_y0p(self, y0p):
        y0min = y0p[0]
        y0max = y0p[1]
        y0disc = y0p[2]
        self.y0s = np.linspace(y0min,y0max,y0disc)

    def set_t0p(self, t0p):
        t0min = t0p[0]
        t0max = t0p[1]
        t0disc = t0p[2]
        self.t0s = np.linspace(t0min,t0max,t0disc)

    def set_vep(self, vep):
        vemin = vep[0]
        vemax = vep[1]
        vedisc = vep[2]
        self.velocities = np.linspace(vemin,vemax,vedisc)

    def set_thp(self, thp):
        thmin = thp[0]
        thmax = thp[1]
        thdisc = thp[2]
        self.thetas = np.deg2rad(np.linspace(thmin,thmax,thdisc))
        
    def set_gap(self, gap):
        gamin = gap[0]
        gamax = gap[1]
        gadisc = gap[2]
        self.gammas = np.deg2rad(np.linspace(gamin,gamax,gadisc))

class Station:
    """
    Station Class
    -------------
    A Station object wraps the relevant data for a seismic station for ease of interfacing with a TrajectoryEstimator
    object.
    """

    def __init__(self, lat, long, elev, ta):
        """
        Initialize a Station object with the relevant positional and arrival time data.
        ______________________________________________________________________________
        lat  : float : degrees latitude  : latitudinal position of the seismic station
        long : float : degrees longitude : longitudinal position of the seismic station
        elev : float : meters            : elevation of the seismic station
        ta   : float : seconds           : arrival time of ballistic wave at seismic station
        ____________________________________________________________________________________
        """
        self.lat = lat
        self.long = long
        self.elev = elev
        self.ta = ta
        self.x = None
        self.y = None
        self.z = None

    def calc_x_y(self, olat, olong):
        """
        Calculate the x (southing) and y (easting) position of the station relative to origin latitude and longitude.
        _____________________________________________________________________________________________
        olat  : float : degrees latitude  : latitudinal position of the origin of the inversion space
        olong : float : degrees longitude : longitudinal position of the origin of the inversion space
        ______________________________________________________________________________________________
        """

        xyz = ECEF_g_to_ECEF_r(np.deg2rad(self.lat),np.deg2rad(self.long),self.elev)

        self.x, self.y, self.z = ECEF_r_to_LTP(xyz,np.deg2rad(olat),np.deg2rad(olong))
        # Convert coordinates to kilometers
        self.x /= 1000
        self.y /= 1000
        self.z /= 1000

class TrajectoryEstimator:
    """
    TrajectoryEstimator Class
    -------------------------
    A TrajectoryEstimator object handles inversion of the seismic data from a meteoroid's ballistic wave to estimate the
    trajectory of the meteoroid through the atmosphere.
    """

    def __init__(self, olat, olong, ot, stations=None):
        """
        Initialize a TrajectoryEstimator object with the origin latitude and longitude, as well as an array of Stations
        _____________________________________________________________________________________________
        olat     : float     : degrees latitude  : latitudinal position of the origin of the inversion space
        olong    : float     : degrees longitude : longitudinal position of the origin of the inversion space
        ot       : TIME      : seconds           : origin time of inversion space
        stations : arraylike : init_val = None   : array containing the seismic stations which will be used for the inversion
        _____________________________________________________________________________________________________________________
        """
        self.olat = olat
        self.olong = olong
        self.otime = ot
        self.stations = stations
        if stations is not None:
            for station in stations:
                station.calc_x_y(olat, olong)

    def set_stations(self, lats, longs, elevs, ts):
        """
        Set the stations array of the TrajectoryEstimator using lists containing station data
        __________________________________________________________________________________________
        lats  : arraylike : degrees latitude  : array containing latitudinal positions of stations
        longs : arraylike : degrees longitude : array containing longitudinal positions of stations
        elevs : arraylike : kilometers        : array containing elevational positions of stations
        ts    : arraylike : seconds           : array containing ballistic wave arrival times of stations
        _________________________________________________________________________________________________ 
        """
        self.stations = np.array([None] * len(lats), dtype=Station)
        for i in range(len(lats)):
            self.stations[i] = Station(lats[i], longs[i], elevs[i], ts[i])
            self.stations[i].calc_x_y(self.olat, self.olong)

    def gridsearch(self,gsp):

        c = 0.32 # velocity of sound in atmosphere (km/s)

        minsq = float('inf') # set initial value for least squares
        optimalTrajectory = Trajectory

        p_track = 0 # percentage tracker
        for theta in gsp.thetas:
            print(str(round(100 * p_track, 2)) + "% complete")
            p_track += 1/gsp.thetas.size
            for gamma in gsp.gammas:
                A = np.array([[np.cos(gamma) * np.sin(theta), np.sin(gamma) * np.sin(theta), -1*np.cos(theta)],
                              [-1*np.sin(gamma),              np.cos(gamma),                 0               ],
                              [np.cos(gamma) * np.cos(theta), np.sin(gamma) * np.cos(theta), np.sin(theta)   ]]) # Rotation matrix for current parameters
                for velocity in gsp.velocities:
                    beta = np.arcsin(c / velocity)
                    for x0 in gsp.x0s:
                        for y0 in gsp.y0s:
                            for t0 in gsp.t0s:
                                sq_sum = 0 # sum of squares of residuals
                                for station in self.stations:
                                    b = np.array([[station.x - x0],
                                                  [station.y - y0],
                                                  [station.z     ]])
                                    X, Y, Z = (A @ b).flatten() # rotated coordinate system with origin at meteoroid-surface intersection and Z along trajectory
    
                                    
                                    ti = t0 + (((np.sqrt(X*X + Y*Y) / np.tan(beta)) - Z) / velocity) # wave arrival time for current parameters
                                    sq_sum += (ti - station.ta) ** 2 # add square of residual
                                    
                                if sq_sum < minsq:
                                    minsq = sq_sum
                                    optimalTrajectory = Trajectory(x0,y0,t0,velocity,theta,gamma,self.olat,self.olong,ssr=sq_sum,time=self.otime) # update optimal trajectory if necessary

        print("100% complete")
        return optimalTrajectory
    
    def invert_data(self, params=None, method="gridsearch"):
        optimalTrajectory = Trajectory
        
        if method == "gridsearch":
            if params is None:
                params = TrajectoryGridsearchParameters()
            optimalTrajectory = self.gridsearch(params)

        return optimalTrajectory

