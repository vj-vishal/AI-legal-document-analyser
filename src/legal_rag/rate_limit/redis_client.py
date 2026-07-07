import redis
import os

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

# Atomically: if current + amount <= limit -> increment and allow
RESERVE_QUOTA_SCRIPT = """
local key = KEYS[1]
local amount = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local current = tonumber(redis.call("GET", key) or "0")

if current + amount > limit then
    return {0, current, limit}
end

local new_val = redis.call("INCRBY", key, amount)
if new_val == amount then
    redis.call("EXPIRE", key, ttl)
end

return {1, new_val, limit}
"""

# Adjust usage after actual cost is known (no limit check, just corrects the count)
ADJUST_QUOTA_SCRIPT = """
local key = KEYS[1]
local delta = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

local new_val = redis.call("INCRBY", key, delta)
if new_val == delta then
    redis.call("EXPIRE", key, ttl)
end
return new_val
"""

reserve_quota = redis_client.register_script(RESERVE_QUOTA_SCRIPT)
adjust_quota = redis_client.register_script(ADJUST_QUOTA_SCRIPT)