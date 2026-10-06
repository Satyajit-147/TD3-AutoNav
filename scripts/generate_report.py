#!/usr/bin/env python3
import os
import json
import numpy as np

def generate_report():
    results_dir = "evaluation/results"
    
    total_episodes = 0
    total_success = 0
    total_collision = 0
    total_timeout = 0
    
    all_success_steps = []
    
    print("| Map | Obstacles | Success Rate | Collision Rate | Timeout Rate | Avg Steps (Success) |")
    print("|:---:|:---------:|:------------:|:--------------:|:------------:|:-------------------:|")
    
    for i in range(10):
        res_file = os.path.join(results_dir, f"results_map_{i}.json")
        if not os.path.exists(res_file):
            print(f"| {i:3d} | - | Error | - | - | - |")
            continue
            
        with open(res_file, 'r') as f:
            res = json.load(f)
            
        map_npy = os.path.join("evaluation/maps", f"map_{i}.npy")
        if os.path.exists(map_npy):
            grid = np.load(map_npy)
            obs_count = int(np.sum(grid))
        else:
            obs_count = "?"
            
        episodes = res['episodes']
        success = res['success']
        collision = res['collision']
        timeout = res['timeout']
        
        succ_rate = (success / episodes) * 100
        coll_rate = (collision / episodes) * 100
        time_rate = (timeout / episodes) * 100
        
        avg_steps = np.mean(res['success_steps']) if res['success_steps'] else 0
        
        print(f"| {i:3d} | {obs_count:9} | {succ_rate:11.1f}% | {coll_rate:13.1f}% | {time_rate:11.1f}% | {avg_steps:18.1f} |")
        
        total_episodes += episodes
        total_success += success
        total_collision += collision
        total_timeout += timeout
        all_success_steps.extend(res['success_steps'])

    print("\n### Final Conclusion\n")
    print(f"- **Total Episodes**: {total_episodes}")
    print(f"- **Overall Success Rate**: {(total_success/total_episodes)*100:.1f}%")
    print(f"- **Overall Collision Rate**: {(total_collision/total_episodes)*100:.1f}%")
    print(f"- **Overall Timeout Rate**: {(total_timeout/total_episodes)*100:.1f}%")
    
    if all_success_steps:
        print(f"- **Average Steps per Success**: {np.mean(all_success_steps):.1f}")
    
    print("\nBased on these results across 10 novel obstacle layouts, the TD3 agent demonstrates...")

if __name__ == '__main__':
    generate_report()
