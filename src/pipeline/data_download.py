import concurrent.futures

from utils.constants import MAX_WORKERS
from utils.copernicus_downloader import regions, process_cams_request, process_era5_region

def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for state, quadrants in regions.items():
            for region_name, region_coords in quadrants.items():
                futures.append(executor.submit(process_cams_request, state, region_name, region_coords))
                futures.append(executor.submit(process_era5_region, state, region_name, region_coords))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Process generated an exception: {exc}")

if __name__ == '__main__':
    main()