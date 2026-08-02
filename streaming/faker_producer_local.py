#!/usr/bin/env python3
import json, time, random, uuid, math
from datetime import datetime, timedelta
from kafka import KafkaProducer

try:
    from faker import Faker
    _fake = Faker('pt_BR')
    def rand_city():
        return _fake.city().lower()
except Exception:
    def rand_city():
        return random.choice(["sao paulo", "rio de janeiro", "salvador",
                              "curitiba", "manaus", "belem", "recife", "fortaleza"])

KAFKA_BROKER   = "localhost:9092"
TOPIC          = "test_olist_orders_stream"
SLEEP_INTERVAL = 0.5

RATE_AMOUNT   = 0.08
RATE_GEO      = 0.06
RATE_OFFHOURS = 0.05
RATE_FREIGHT  = 0.04
REPEAT_RATE   = 0.45

STATES = ["SP", "RJ", "MG", "RS", "PR", "BA", "PE", "CE", "AM", "PA", "AC", "RR", "AP"]
FAR_STATE_PAIRS = [("SP","AM"),("SP","PA"),("RJ","AM"),("MG","AC"),
                   ("SP","RR"),("PR","AP"),("RS","AM"),("SP","AC")]
CATS = ["electronics_accessories","mobile_accessories","smart_home_devices",
        "fitness_technology","gaming_peripherals","portable_batteries",
        "computers_accessories","telephony","home_appliances","sports_leisure"]

REPEAT_POOL = [uuid.uuid4().hex[:32] for _ in range(15)]
cust_last_state = {}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

print("Starting local producer -> %s, topic '%s'" % (KAFKA_BROKER, TOPIC))

while True:
    try:
        if random.random() < REPEAT_RATE:
            customer_unique_id = random.choice(REPEAT_POOL)
        else:
            customer_unique_id = uuid.uuid4().hex[:32]
        customer_id = uuid.uuid4().hex[:32]
        state = random.choice(STATES)
        city = rand_city()

        order_id = uuid.uuid4().hex[:32]
        status = random.choices(["delivered","shipped","canceled","invoiced"],
                                weights=[0.7,0.15,0.1,0.05])[0]

        now = datetime.utcnow()
        inject_offhours = random.random() < RATE_OFFHOURS
        if inject_offhours:
            purchase_ts = now.replace(hour=random.randint(1, 4),
                                      minute=random.randint(0, 59))
        else:
            purchase_ts = now
        approved_ts  = purchase_ts + timedelta(hours=random.randint(1, 12))
        carrier_ts   = approved_ts + timedelta(days=random.randint(1, 3))
        delivered_ts = carrier_ts + timedelta(days=random.randint(2, 15))
        estimated_ts = purchase_ts + timedelta(days=random.randint(7, 25))

        num_items = random.choices([1,2,3], weights=[0.7,0.2,0.1])[0]
        items = []
        for i in range(num_items):
            price = round(max(5.0, math.exp(random.gauss(4.0, 1.0))), 2)
            freight = round(random.uniform(5, 30), 2)
            items.append({
                "order_item_id": i + 1,
                "product_id": uuid.uuid4().hex[:32],
                "seller_id": uuid.uuid4().hex[:32],
                "price": price,
                "freight_value": freight,
                "product_category": random.choice(CATS),
            })

        if random.random() < RATE_AMOUNT:
            items[0]["price"] = round(random.uniform(3500, 20000), 2)
        if inject_offhours:
            items[0]["price"] = max(items[0]["price"], round(random.uniform(1200, 4000), 2))
        if random.random() < RATE_FREIGHT:
            items[0]["freight_value"] = round(random.uniform(150, 400), 2)

        total_value = sum(it["price"] + it["freight_value"] for it in items)

        is_geo_anomaly = False
        if random.random() < RATE_GEO and customer_unique_id in cust_last_state:
            last_state = cust_last_state[customer_unique_id]
            far = [b for a, b in FAR_STATE_PAIRS if a == last_state] + \
                  [a for a, b in FAR_STATE_PAIRS if b == last_state]
            if far:
                state = random.choice(far)
                is_geo_anomaly = True
        cust_last_state[customer_unique_id] = state

        def fmt(ts):
            return ts.strftime("%d-%m-%Y %H:%M")

        order = {
            "order_id": order_id,
            "customer_id": customer_id,
            "customer_unique_id": customer_unique_id,
            "customer_state": state,
            "customer_city": city,
            "order_status": status,
            "order_purchase_timestamp": fmt(purchase_ts),
            "order_approved_at": fmt(approved_ts) if status != "canceled" else "",
            "order_delivered_carrier_date": fmt(carrier_ts) if status == "delivered" else "",
            "order_delivered_customer_date": fmt(delivered_ts) if status == "delivered" else "",
            "order_estimated_delivery_date": fmt(estimated_ts),
            "items": items,
            "payment_type": random.choice(["credit_card","boleto","voucher","debit_card"]),
            "payment_installments": random.randint(1, 12),
            "total_payment_value": round(total_value, 2),
            "injected_anomaly": {
                "amount_anomaly": str(items[0]["price"] > 1000),
                "geo_anomaly": str(is_geo_anomaly),
            },
        }

        producer.send(TOPIC, value=order)
        print("Sent %s | total=%8.2f | state=%2s | geo=%s" % (order_id[:8], total_value, state, is_geo_anomaly))
    except Exception as e:
        print("Error: %s" % e)

    time.sleep(SLEEP_INTERVAL)
