#!/usr/bin/env python3
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

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

# Function to calculate normalized skew ratio and anticipated price range
def calculate_skew_ratio_and_price_range(chain, stock_price, days_to_expiry):
    left_side = chain[chain['strike'] < stock_price]
    right_side = chain[chain['strike'] > stock_price]
    
    left_avg_iv = left_side['impliedVolatility'].mean() if not left_side.empty else 0
    right_avg_iv = right_side['impliedVolatility'].mean() if not right_side.empty else 0
    
    # Normalize the skew ratio to be between -1 and 1
    skew_ratio = (left_avg_iv - right_avg_iv) / (left_avg_iv + right_avg_iv) if (left_avg_iv + right_avg_iv) != 0 else 0
    
    # Adjust for zero days to expiry
    days_to_expiry = max(days_to_expiry, 1)
    
    # Annualize the implied volatilities
    annual_factor = np.sqrt(252 / days_to_expiry)
    left_iv_annualized = left_avg_iv * annual_factor
    right_iv_annualized = right_avg_iv * annual_factor
    
    # Calculate anticipated price range
    left_price_range = stock_price * np.exp(left_iv_annualized)
    right_price_range = stock_price * np.exp(right_iv_annualized)
    
    return skew_ratio, left_price_range, right_price_range, left_avg_iv, right_avg_iv

# Function to calculate the put/call volume ratio
def calculate_put_call_volume_ratio(chain):
    put_volume = chain[chain['optionType'] == "put"]['volume'].sum()
    call_volume = chain[chain['optionType'] == "call"]['volume'].sum()
    # Reverse the ratio for bullish/bearish indication
    return (call_volume - put_volume) / (put_volume + call_volume) if (put_volume + call_volume) != 0 else 0

# Function to calculate the open interest ratio
def calculate_open_interest_ratio(chain):
    put_open_interest = chain[chain['optionType'] == "put"]['openInterest'].sum()
    call_open_interest = chain[chain['optionType'] == "call"]['openInterest'].sum()
    # Reverse the ratio for bullish/bearish indication
    return (call_open_interest - put_open_interest) / (put_open_interest + call_open_interest) if (put_open_interest + call_open_interest) != 0 else 0

# Main function
def main(ticker):
    options = option_chains(ticker)
    stock_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    
    expirations = options['expiration'].unique()
    skew_ratios = []
    volume_ratios = []
    open_interest_ratios = []
    
    for expiry in expirations:
        days_to_expiry = (expiry - pd.Timestamp.today()).days
        days_to_expiry = max(days_to_expiry, 1)  # Ensure days_to_expiry is at least 1
        options_at_expiry = options[options["expiration"] == expiry]
        
        calls_at_expiry = options_at_expiry[options_at_expiry["optionType"] == "call"]
        puts_at_expiry = options_at_expiry[options_at_expiry["optionType"] == "put"]
        
        # Filter out zero implied volatility
        filtered_calls_at_expiry = calls_at_expiry[calls_at_expiry.impliedVolatility >= 0.001]
        filtered_puts_at_expiry = puts_at_expiry[puts_at_expiry.impliedVolatility >= 0.001]
        
        # Calculate skew ratios and anticipated price ranges for calls and puts
        call_skew_ratio, _, _, call_left_avg_iv, call_right_avg_iv = calculate_skew_ratio_and_price_range(filtered_calls_at_expiry, stock_price, days_to_expiry)
        put_skew_ratio, _, _, put_left_avg_iv, put_right_avg_iv = calculate_skew_ratio_and_price_range(filtered_puts_at_expiry, stock_price, days_to_expiry)
        
        # Calculate the norm and deviations
        avg_iv_norm = (call_left_avg_iv + call_right_avg_iv + put_left_avg_iv + put_right_avg_iv) / 4
        call_deviation = abs((call_left_avg_iv + call_right_avg_iv) / 2 - avg_iv_norm)
        put_deviation = abs((put_left_avg_iv + put_right_avg_iv) / 2 - avg_iv_norm)
        
        # Calculate weights based on deviations
        total_deviation = call_deviation + put_deviation
        call_weight = call_deviation / total_deviation if total_deviation != 0 else 0.5
        put_weight = put_deviation / total_deviation if total_deviation != 0 else 0.5
        
        # Calculate the weighted average skew ratio for this expiration
        weighted_avg_skew_ratio = call_weight * call_skew_ratio + put_weight * put_skew_ratio
        skew_ratios.append(weighted_avg_skew_ratio)
        
        # Calculate volume ratio and open interest ratio
        volume_ratio = calculate_put_call_volume_ratio(options_at_expiry)
        volume_ratios.append(volume_ratio)
        
        open_interest_ratio = calculate_open_interest_ratio(options_at_expiry)
        open_interest_ratios.append(open_interest_ratio)
    
    # Calculate the overall skew shift ratio
    skew_shift_ratio = np.mean(skew_ratios)
    avg_volume_ratio = np.mean(volume_ratios)
    avg_open_interest_ratio = np.mean(open_interest_ratios)
    
    # Print the results
    print(f"Stock Price: {stock_price:.2f}")
    
    print("Skew Shift Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Skew Shift Ratio: {skew_ratios[i]:.2f}")
        
    print(f"Average Skew Shift Ratio: {skew_shift_ratio:.2f}\n")
    
    print("Put/Call Volume Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Put/Call Volume Ratio: {volume_ratios[i]:.2f}")
        
    print(f"Average Put/Call Volume Ratio: {avg_volume_ratio:.2f}\n")
    
    print("Open Interest Ratios:")
    for i, expiry in enumerate(expirations):
        print(f"Expiration: {expiry.strftime('%Y-%m-%d')}, Open Interest Ratio: {open_interest_ratios[i]:.2f}")
        
    print(f"Average Open Interest Ratio: {avg_open_interest_ratio:.2f}")

main("RKLB")
