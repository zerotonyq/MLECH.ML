WITH car_stats AS (
    SELECT car_id,
       AVG(rating) AS avg_trip_rating,
       AVG(speed_avg) AS avg_car_speed,
       SUM(stop_times) AS sum_of_stops,
       AVG(ride_duration) AS avg_ride_duration,
       SUM(distance) AS sum_of_distance,
       AVG(user_rating) AS avg_user_rating,
       AVG(user_time_accident) AS avg_users_accidents
    FROM rides_info rd LEFT JOIN driver_info di USING(user_id)
    WHERE rd.distance < 7000 AND
          rd.ride_duration < 300 AND
          rd.ride_cost < 4000
    GROUP BY car_id
)

SELECT *
FROM car_test JOIN car_stats USING(car_id);
