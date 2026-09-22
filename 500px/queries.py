"""500px public website GraphQL selections, verified on 2026-09-11.

Source: the site's getPhotoById.generated and gallery component documents.
Image URLs are signed renditions; preserve the complete values verbatim.
"""

ENDPOINT = "https://api-neo.500px.com/graphql"

PHOTO_FRAGMENT = """
fragment PhotoMetadata on Photo {
  __typename
  id title description uploadedAt uploadedLocation takenAt
  width height dominantColorLight dominantColorDark
  urls { size_600 size_1024 size_2048 size_4k }
  uploader {
    id username displayName avatar location city country state
    membership { membership }
  }
  location locationText camera lens aperture focalLength shutterSpeed iso
  category techniques
  downloadable isNsfw isPrivate isDeleted isInReview needLoginToView
  pulseScore viewCount likeCount favorCount commentCount shareCount repostCount
  hasComment aiArtAnalysis taggedAigc userDeclaredAigc
  geminiDetail { category style technique title keyword }
  honors {
    __typename
    ... on PhotoHonorSelected { type }
    ... on PhotoHonorAmbassadorsPick { ambassador { id avatar } }
    ... on PhotoHonorPxGallery { gallery { id name } }
  }
}
"""

GALLERY_QUERY = """
query GalleryData($galleryId: ID!, $first: Int!, $after: String) {
  getGalleryById(id: $galleryId) {
    id name description kind itemCount updatedAt
    isPrivate isNsfw isDeleted
    likeCount viewCount commentCount shareCount repostCount pulseScore
    creator { id username displayName avatar }
  }
  pageGalleryItems(galleryId: $galleryId, first: $first, after: $after) {
    edges {
      cursor
      node {
        __typename
        ...PhotoMetadata
        ... on PhotoGroup {
          id title description numCount publicItemCount
          isDeleted isPrivate isInReview isNsfw
        }
        ... on Video { id title isDeleted isPrivate isInReview isNsfw }
      }
    }
    pageInfo { endCursor hasNextPage hasPreviousPage }
  }
}
""" + PHOTO_FRAGMENT

PHOTO_QUERY = """
query PhotoData($id: ID!) {
  getPhotoById(id: $id) { ...PhotoMetadata }
}
""" + PHOTO_FRAGMENT

# The website returns the complete member array here, without cursor arguments.
GROUP_QUERY = """
query GroupPhotos($groupId: ID!) {
  getPhotosByGroupId(groupId: $groupId) { ...PhotoMetadata }
}
""" + PHOTO_FRAGMENT
