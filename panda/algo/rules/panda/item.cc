/*----------------------------------------------------------------------
  File    : item.cpp
  Contents: itemset management
  Author  : Bart Goethals
  Update  : 04/04/2003
  ----------------------------------------------------------------------*/

#include <stdio.h>
#include "item.hh"

Item_::Item_()
{
  supp = 0;
  parent = 0;
  nodelink = 0;
  id = 0;
  children = 0;
  flag = 0;
}

Item_::~Item_()
{}

Item::Item(int s, Item *p)
{
  item = new Item_();
  item->id = s;
  item->parent = p;
}

Item::Item(const Item& i)
{
  item = new Item_();
  item->id       = i.getId(); //tmp->id;
  item->parent   = i.getParent(); //tmp->parent;
  item->children = i.getChildren(); //tmp->children;
  item->nodelink = i.getNext(); //tmp->nodelink;
  item->supp     = i.getSupport(); //tmp->supp;
  item->flag     = i.getFlag();
}

Item::~Item()
{
  delete item;
}

set<Item> *Item::makeChildren() const
{
  if(item->children==0) item->children = new set<Item>;
  return item->children;
}


void Item::removeChildren() const
{
  set<Item> *items = item->children;
  for(set<Item>::iterator it = items->begin();it != items->end(); it++) it->removeChildren();
  delete item->children;
  item->children = 0;
}
