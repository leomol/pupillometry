# 2020-02-18. Leonardo Molina.
# 2021-01-23. Last modified.

class Subscription:
    def __init__(self, publisher):
        self.publisher = publisher
    
    def unsubscribe(self):
        self.publisher.unsubscribe(self);

class Publisher:
    
    def __init__(self):
        self.__subscriptionToEvent = {}
        self.__eventToSubscriptions = {}
        self.__subscriptionToCallback = {}
        pass
    
    def contains(self, event):
        return event in self.__eventToSubscriptions
    
    # subscription = subscribe(callback, <event>)
    def subscribe(self, callback, event = ""):
        subscription = Subscription(self)
        if event not in self.__eventToSubscriptions:
            self.__eventToSubscriptions[event] = []
        self.__eventToSubscriptions[event].append(subscription)
        self.__subscriptionToCallback[subscription] = callback
        self.__subscriptionToEvent[subscription] = event
        return subscription
    
    # unsubscribe(subscription)
    def unsubscribe(self, subscription):
        if subscription in self.__subscriptionToEvent:
            event = self.__subscriptionToEvent[subscription]
            self.__eventToSubscriptions[event].remove(subscription)
            del self.__subscriptionToEvent[subscription]
            del self.__subscriptionToCallback[subscription]
    
    # invoke(<event>)
    def invoke(self, event = "", *args, **kargs):
        if event in self.__eventToSubscriptions:
            for subscription in self.__eventToSubscriptions[event]:
                callback = self.__subscriptionToCallback[subscription]
                callback(*args, **kargs)
            
# publisher = Publisher()
# subscription1 = publisher.subscribe(lambda : print('Called 1!'), 'EventA')
# subscription2 = publisher.subscribe(lambda : print('Called 2!'), 'EventA')
# subscription3 = publisher.subscribe(lambda message : print(message), 'EventB')
# subscription1.unsubscribe()
# publisher.invoke('EventA')
# publisher.invoke('EventB', 'Hello!')