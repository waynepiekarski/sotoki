rm memory.log || echo "fresh start"
while true; do
    ps -C sotoki -o pid=,%mem=,vsz= >> memory.log
    sleep 1
done
