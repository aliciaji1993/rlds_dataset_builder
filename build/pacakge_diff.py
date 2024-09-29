# requirements from rlds package
req = open("requirements.txt").readlines()
# installed packages from docker image after PrismaticVLM environment setup
curr = open("current.txt").readlines()

req_packages = dict(
    [line.strip("\n").split("==") for line in filter(lambda x: "@" not in x, req)]
)
curr_packages = dict(
    [line.strip("\n").split("==") for line in filter(lambda x: "@" not in x, curr)]
)


print("======== Conflicting packages ========")
for pacakge, version in req_packages.items():
    if pacakge in curr_packages:
        print(
            f"Package: {pacakge}, current: {curr_packages[pacakge]}, required: {req_packages[pacakge]}"
        )
