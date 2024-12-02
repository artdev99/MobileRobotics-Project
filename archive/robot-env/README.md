# To create envs
1. open [draw.io](https://app.diagrams.net/)
2. Load locally `robot-env.draw.io`
3. Modify
4. Saves As > Type PNG > Where Download (to save the png)
5. Optionally you can also save the XML `.drawio`, if you want to create a new scene and use it as a base
6. For print purposes you can convert the `png` to `pdf` using online solutions like [pdf24](https://tools.pdf24.org/en/png-to-pdf) or run `png2pdf.py` (requires `PIL` and `img2pdf`)
    
# Color code
- Background: black
- global obstacles: red
- goal: green
- robot/start: white

Page format: A3

# draw.io
- Diagram Menu
    - Background Color: Black
    - Paper Size: A3
    - Paper Orientation: Landscape
- View Menu
    - Units
        - Milimeters
- Select Delay
    - Arange Menu
    - Size 36mm 36mm

Using delay shape is a good approx for thymio robot: <br>
![Approx](https://i.postimg.cc/bw3hj8wb/image-2024-11-13-100740839.png)

# File name convention
- `rx.*` robot present, real life purposes
- `snx.*` where no robot is present, simulation purposes
- `srx.*` where robot is present, simulation purposes
