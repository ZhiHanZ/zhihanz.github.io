import rss from '@astrojs/rss'
import { getCollection } from 'astro:content'
import { loadSite } from '../lib/site'
import type { APIContext } from 'astro'

export async function GET(context: APIContext) {
  const site = loadSite()

  const posts = await getCollection('posts')
  const thoughts = await getCollection('thoughts')

  const allPosts = [...posts, ...thoughts]
    .map(post => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description || '',
      link: `/${post.collection}/${post.id}/`
    }))
    .sort((a, b) => b.pubDate.getTime() - a.pubDate.getTime())

  return rss({
    title: site.title,
    description: site.description,
    site: context.site || site.baseURL,
    items: allPosts,
    customData: `<language>en-us</language>`
  })
}
