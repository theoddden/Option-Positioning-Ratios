#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.stats import norm

def option_chains(ticker):
    asset = yf.Ticker(ticker)
    expirations = asset.options
    chains = pd.DataFrame()
    for expiration in expirations:
        opt = asset.option_chain(expiration)
        calls = opt.calls
        calls['optionType'] = "call"
        puts = opt.puts
        puts['optionType'] = "put"
        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        chains = pd.concat([chains, chain])
    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
    return chains

def calculate_skew_ratio_and_price_range(chain, stock_price, days_to_expiry):
    left_side = chain[chain['strike'] < stock_price]
    right_side = chain[chain['strike'] > stock_price]
    
    left_avg_iv = left_side['impliedVolatility'].mean() if not left_side.empty else 0
    right_avg_iv = right_side['impliedVolatility'].mean() if not right_side.empty else 0
    
    skew_ratio = (left_avg_iv - right_avg_iv) / (left_avg_iv + right_avg_iv) if (left_avg_iv + right_avg_iv) != 0 else 0
    
    days_to_expiry = max(days_to_expiry, 1)
    
    annual_factor = np.sqrt(252 / days_to_expiry)
    left_iv_annualized = left_avg_iv * annual_factor
    right_iv_annualized = right_avg_iv * annual_factor
    
    left_price_range = stock_price * np.exp(left_iv_annualized)
    right_price_range = stock_price * np.exp(right_iv_annualized)
    
    return skew_ratio, left_price_range, right_price_range, left_avg_iv, right_avg_iv

def calculate_put_call_volume_ratio(chain):
    put_volume = chain[chain['optionType'] == "put"]['volume'].sum()
    call_volume = chain[chain['optionType'] == "call"]['volume'].sum()
    return (call_volume - put_volume) / (put_volume + call_volume) if (put_volume + call_volume) != 0 else 0

def calculate_open_interest_ratio(chain):
    put_open_interest = chain[chain['optionType'] == "put"]['openInterest'].sum()
    call_open_interest = chain[chain['optionType'] == "call"]['openInterest'].sum()
    return (call_open_interest - put_open_interest) / (put_open_interest + call_open_interest) if (put_open_interest + call_open_interest) != 0 else 0

def estimate_delta(row, spot_price, risk_free_rate=0.05):
    T = row['daysToExpiration'] / 365.0
    K = row['strike']
    sigma = row['impliedVolatility']
    
    if sigma == 0:
        return 0
    
    d1 = (np.log(spot_price / K) + (risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if row['optionType'] == 'call':
        return norm.cdf(d1)
    else:  # put
        return norm.cdf(d1) - 1

def estimate_gamma(row, spot_price, risk_free_rate=0.05):
    S = spot_price
    K = row['strike']
    T = row['daysToExpiration'] / 365.0
    r = risk_free_rate
    sigma = row['impliedVolatility']
    
    if sigma == 0 or T == 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma

def calculate_delta_exposure_ratio(chain, spot_price):
    chain['delta'] = chain.apply(lambda row: estimate_delta(row, spot_price), axis=1)
    chain['delta_exposure'] = chain['delta'] * chain['openInterest']
    put_delta_exposure = chain[chain['optionType'] == "put"]['delta_exposure'].sum()
    call_delta_exposure = chain[chain['optionType'] == "call"]['delta_exposure'].sum()
    total_exposure = abs(put_delta_exposure) + abs(call_delta_exposure)
    return (call_delta_exposure + put_delta_exposure) / total_exposure if total_exposure != 0 else 0

def calculate_gamma_exposure_ratio(chain, spot_price):
    chain['gamma'] = chain.apply(lambda row: estimate_gamma(row, spot_price), axis=1)
    chain['gamma_exposure'] = chain['gamma'] * chain['openInterest'] * spot_price
    put_gamma_exposure = chain[chain['optionType'] == "put"]['gamma_exposure'].sum()
    call_gamma_exposure = chain[chain['optionType'] == "call"]['gamma_exposure'].sum()
    total_exposure = put_gamma_exposure + call_gamma_exposure
    return (call_gamma_exposure - put_gamma_exposure) / total_exposure if total_exposure != 0 else 0

def main(ticker):
    options = option_chains(ticker)
    stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    
    expirations = options['expiration'].unique()
    skew_ratios = []
    volume_ratios = []
    open_interest_ratios = []
    delta_exposure_ratios = []
    gamma_exposure_ratios = []
    
    for expiry in expirations:
        days_to_expiry = (expiry - pd.Timestamp.today()).days
        days_to_expiry = max(days_to_expiry, 1)
        options_at_expiry = options[options["expiration"] == expiry]
        
        calls_at_expiry = options_at_expiry[options_at_expiry["optionType"] == "call"]
        puts_at_expiry = options_at_expiry[options_at_expiry["optionType"] == "put"]
        
        filtered_calls_at_expiry = calls_at_expiry[calls_at_expiry.impliedVolatility >= 0.001]
        filtered_puts_at_expiry = puts_at_expiry[puts_at_expiry.impliedVolatility >= 0.001]
        
        call_skew_ratio, _, _, call_left_avg_iv, call_right_avg_iv = calculate_skew_ratio_and_price_range(filtered_calls_at_expiry, stock_price, days_to_expiry)
        put_skew_ratio, _, _, put_left_avg_iv, put_right_avg_iv = calculate_skew_ratio_and_price_range(filtered_puts_at_expiry, stock_price, days_to_expiry)
        
        avg_iv_norm = (call_left_avg_iv + call_right_avg_iv + put_left_avg_iv + put_right_avg_iv) / 4
        call_deviation = abs((call_left_avg_iv + call_right_avg_iv) / 2 - avg_iv_norm)
        put_deviation = abs((put_left_avg_iv + put_right_avg_iv) / 2 - avg_iv_norm)
        
        total_deviation = call_deviation + put_deviation
        call_weight = call_deviation / total_deviation if total_deviation != 0 else 0.5
        put_weight = put_deviation / total_deviation if total_deviation != 0 else 0.5
        
        weighted_avg_skew_ratio = call_weight * call_skew_ratio + put_weight * put_skew_ratio
        skew_ratios.append(weighted_avg_skew_ratio)
        
        volume_ratio = calculate_put_call_volume_ratio(options_at_expiry)
        volume_ratios.append(volume_ratio)
        
        open_interest_ratio = calculate_open_interest_ratio(options_at_expiry)
        open_interest_ratios.append(open_interest_ratio)
        
        delta_exposure_ratio = calculate_delta_exposure_ratio(options_at_expiry, stock_price)
        delta_exposure_ratios.append(delta_exposure_ratio)
        
        gamma_exposure_ratio = calculate_gamma_exposure_ratio(options_at_expiry, stock_price)
        gamma_exposure_ratios.append(gamma_exposure_ratio)
        
    skew_shift_ratio = np.mean(skew_ratios)
    avg_volume_ratio = np.mean(volume_ratios)
    avg_open_interest_ratio = np.mean(open_interest_ratios)
    avg_delta_exposure_ratio = np.mean(delta_exposure_ratios)
    avg_gamma_exposure_ratio = np.mean(gamma_exposure_ratios)
    
    print(f"Stock Price: {stock_price:.2f}")
    
    print("\nSkew Shift Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Skew Shift Ratio: {skew_ratios[i]:.2f}")
    print(f"Average Skew Shift Ratio: {skew_shift_ratio:.2f}")
    
    print("\nPut/Call Volume Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Put/Call Volume Ratio: {volume_ratios[i]:.2f}")
    print(f"Average Put/Call Volume Ratio: {avg_volume_ratio:.2f}")
    
    print("\nOpen Interest Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Open Interest Ratio: {open_interest_ratios[i]:.2f}")
    print(f"Average Open Interest Ratio: {avg_open_interest_ratio:.2f}")
    
    print("\nDelta Exposure Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Delta Exposure Ratio: {delta_exposure_ratios[i]:.2f}")
    print(f"Average Delta Exposure Ratio: {avg_delta_exposure_ratio:.2f}")
    
    print("\nGamma Exposure Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Gamma Exposure Ratio: {gamma_exposure_ratios[i]:.2f}")
    print(f"Average Gamma Exposure Ratio: {avg_gamma_exposure_ratio:.2f}")
    
if __name__ == "__main__":
    main("RKLB")