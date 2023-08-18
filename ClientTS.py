# class Client

class ClientTS(object):
    """
    Container that stores variables that are relevant to customer at certain 
    period of time (time-series state)
    """
    
    # dFirstEvent
    # dLastEvent
    # dThirdEvent
    # periods
    # frequency
    # recency
    # dLastEventState
    # recencyState
    # recencyTS
    # dBirthState
    # loyalty
    # loyaltyState
    # loyaltyTS
    # T
    # T_State
    # T_TS
    # D
    # D_State
    # D_TS    
    # C
    # C_Orig
    # status
    # dDeath
    # dDeathState
    # dDeathObserved
    # tRemainingObserved
    # tRemainingEstimate
    # ratePoisson
    # pPoisson
    # tPoissonLifetime
    # tPoissonDeath
    # moneySum
    # moneyMedian
    # moneyDaily
    # moneyDailyStep

    # primary ranks
    # r10_R
    # r10_F
    # r10_L
    # r10_C
    # r10_C_Orig
    # r10_M # moneyDaily
    
    # composite ranks
    # r10_MC
    # r10_FM
    # r10_FC
    # r10_FMC
    # r10_LFM
    # r10_LFMC
    # r10_RF
    # r10_RFM
    # r10_RFMC
    
    # Long and short Trends    
        # 'trend_r10_MC', 'trend_r10_FM', 'trend_r10_FC', 'trend_r10_FMC', 'trend_r10_LFMC', 
        # 'trend_short_r10_MC', 'trend_short_r10_FM', 'trend_short_r10_FC', 'trend_short_r10_FMC', 
        # 'trend_short_r10_LFMC', 'trend_r10_RF', 'trend_r10_RFM', 'trend_r10_RFMC'
        # 'trend_short_r10_RF', 'trend_short_r10_RFM', 'trend_short_r10_RFMC'
    
    
    def __init__(self, clientId):
    
        self.id = clientId
        self.descriptors = {'pChurn': 0}
        self.stats = None # instance of Statistics
        
    def show(self):        
        print('id : {}'.format(self.id))
        print('descriptors:')
        for key, value in self.descriptors.items():
            print('{} : {}'.format(key, value))
        return
        
    # def __repr__(self):
    #     if self.keys():
    #         m = max(map(len, list(self.keys()))) + 1
    #         return ''.join([k.rjust(m) + ': ' + repr(v)
    #                           for k, v in self.items()])
    #     else:
    #         return self.__class__.__name__ + "()"

    # def __dir__(self):
    #     return list(self.keys())
    