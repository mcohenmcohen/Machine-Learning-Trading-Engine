from stock_system import IQFeed

feed = IQFeed.DataFeed()

# Set protocol - will return a one-liner back
# print feed.send_msg('S,SET PROTOCOL,5.2')

# print feed.send_msg('S,REQUEST WATCHES', port=5009)
# print feed.send_msg('S,UNWATCH ALL', port=5009)

# Real time test
# print feed.send_msg('S,REQUEST CURRENT UPDATE FIELDNAMES', port=5009)
# print feed.send_msg('S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol', port=5009)
# print feed.send_msg('S,REQUEST CURRENT UPDATE FIELDNAMES', port=5009)
# print feed.get_data_stream('FB')

# Historical data
# 60 - minute data
# 3600 - hourly data
# print feed.send_msg('HIT,FB,3600,20170101 075000,,,093000,160000,1\n')
# Daily data
print feed.send_msg('HDX,SPY,1000,1\n')

# News test
# print feed.send_msg('NHL')
# print feed.send_msg('NHL', port=5009)   # Doesn't seem to work
# print feed.send_msg('NHL,,TSLA')  # XML is default
# print feed.send_msg('NHL,,PM:KO,t')  # t - text

# News story by ID
# print feed.send_msg('NSY,21886020296')

# News story count
# print feed.send_msg('NSC,TSLA,,,20170508')

# Search by filter
# print feed.send_msg('SBF,d,FACE')
