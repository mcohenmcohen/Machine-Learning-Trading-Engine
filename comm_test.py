from stock_system import IQFeed, DBUtils

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

# News test
#print feed.send_msg('NHL')
# print feed.send_msg('NHL', port=5009)   # Doesn't seem to work
#print feed.send_msg('NHL,,TSLA')  # XML is default
#print feed.send_msg('NHL,,PM:KO,t')  # t - text

# News story by ID
#print feed.send_msg('NSY,21886020296')

# News story count
print feed.send_msg('NSC,TSLA,,,20170508')
