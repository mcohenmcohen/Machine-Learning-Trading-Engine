from stock_system import IQFeed

feed = IQFeed.DataFeed()

# Set protocol - will return a one-liner back
# print feed.get_data('S,SET PROTOCOL,5.2')

# print feed.get_data('S,REQUEST WATCHES', port=5009)
# print feed.get_data('S,UNWATCH ALL', port=5009)

# Real time test
# print feed.get_data('S,REQUEST CURRENT UPDATE FIELDNAMES', port=5009)
# print feed.get_data('S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol', port=5009)
# print feed.get_data('S,REQUEST CURRENT UPDATE FIELDNAMES', port=5009)
# print feed.get_data_stream('FB')

# Historical data
# 60 - minute data
# 3600 - hourly data
# print feed.get_data('HIT,FB,3600,20170101 075000,,,093000,160000,1\n')
# Daily data
print feed.get_data('HDX,SPY,1000,1\n')

# News test
# print feed.get_data('NHL')
# print feed.get_data('NHL', port=5009)   # Doesn't seem to work
# print feed.get_data('NHL,,TSLA')  # XML is default
# print feed.get_data('NHL,,PM:KO,t')  # t - text

# News story by ID
# print feed.get_data('NSY,21886020296')

# News story count
# print feed.get_data('NSC,TSLA,,,20170508')

# Search by filter
# print feed.get_data('SBF,d,FACE')

# Options chain
print feed.get_data('CEO,GOOG,pc,,1')
