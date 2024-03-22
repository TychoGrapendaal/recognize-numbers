class Button {
  int x,y,w,h;
  color c;
  String text;
  
  public Button(int x, int y, int w, int h, color c, String text) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.c = c;
    this.text = text;
  }
  
  public void show() {
    strokeWeight(0);
    fill(c);
    rect(x,y,w,h);
    fill(0);
    textSize((w+h)/4);
    textAlign(CENTER, CENTER);
    text(text, x+w/2.0, y+h/2.0);
  }
  
  public boolean hit(int x, int y) {
    return (x >= this.x && x <= this.x+this.w && 
      y >= this.y && y <= this.y+this.h);
  }
}
