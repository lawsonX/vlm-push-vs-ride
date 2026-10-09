#!/bin/bash
GW=$(ip route | awk '/default/ {print $3; exit}')
echo "gateway: $GW"
for p in 7890 7897 1080 8889 10808 10809; do
  code=$(curl -s -m 5 -x "http://$GW:$p" https://github.com -o /dev/null -w "%{http_code}" 2>/dev/null)
  echo "$GW:$p -> $code"
done
echo --- gh_login2 log:
cat /home/lawson/vlm-active/logs/gh_login2.log
