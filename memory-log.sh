while true; do
    ps -C sotoki -o pid=,%mem=,vsz= >> /tmp/mem.log
    sleep 1
done
