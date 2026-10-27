import sys; sys.path.insert(0, "scripts")

def price_binomial_tree(option_type, option_style, spot, strike, expiry, volatility, rate, steps, exercise_dates=None):
    if expiry <= 0: return max(spot-strike,0) if option_type=="call" else max(strike-spot,0)
    dt = expiry/steps
    u = pow(2.718281828459045, volatility*(expiry/steps)**0.5) if False else None
    import math
    u = math.exp(volatility*math.sqrt(dt)); d = 1/u
    p = 1.0 if math.isclose(u,d) else (math.exp(rate*dt)-d)/(u-d)
    discount = math.exp(-rate*dt)
    values = []
    for downs in range(steps+1):
        ups = steps-downs
        price = spot*(u**ups)*(d**downs)
        values.append(max(price-strike,0) if option_type=="call" else max(strike-price,0))
    exercise = set(range(steps)) if option_style=="american" else (set(x for x in exercise_dates if x<steps) if option_style=="bermudan" and exercise_dates else set())
    for step in range(steps-1,-1,-1):
        cont = [(p*values[i]+(1-p)*values[i+1])*discount for i in range(step+1)]
        if step in exercise:
            for i in range(step+1):
                ups = step-i
                price = spot*(u**ups)*(d**i)
                ex = max(price-strike,0) if option_type=="call" else max(strike-price,0)
                values[i] = max(ex, cont[i])
        else:
            values = cont + values[step+1:]
    return values[0]

print("euro call", repr(price_binomial_tree("call","european",100,100,1,0.2,0.05,100)))
print("euro put", repr(price_binomial_tree("put","european",100,100,1,0.2,0.05,100)))
print("amer put", repr(price_binomial_tree("put","american",100,100,1,0.2,0.05,100)))
print("berm put", repr(price_binomial_tree("put","bermudan",100,100,1,0.2,0.05,100,list(range(0,100,10)))))
