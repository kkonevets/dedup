import io
import json
import tools


with io.open('../data/release/1cnrel-2019-01-09-17-2019-02-04-09.json',
             encoding='utf8') as f:
    feed = json.load(f)

tools.feed2mongo(feed, 'release')
